import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
import wandb
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    # Trial mode options
    parser.add_argument('--trial', action='store_true', help='Enable trial mode for quick testing')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--max_iters', type=int, default=None, help='Override max iterations')
    parser.add_argument('--val_freq', type=int, default=None, help='Override validation frequency')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='Override num workers')
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Trial mode: override config
    if args.trial:
        if args.max_iters is not None:
            config.train.max_iters = args.max_iters
        if args.val_freq is not None:
            config.train.val_freq = args.val_freq
        if args.batch_size is not None:
            config.train.batch_size = args.batch_size
        if args.num_workers is not None:
            config.train.num_workers = args.num_workers
        # Use smaller report interval in trial mode
        if args.train_report_iter == 200:  # default value
            args.train_report_iter = 50

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    
    # Initialize wandb (skip if --no_wandb or --trial)
    use_wandb = not args.no_wandb and not args.trial
    if use_wandb:
        wandb.init(
            project="tagmol",
            name=f"{config_name}_{args.tag}" if args.tag else config_name,
            config=config,
            dir=log_dir,
            save_code=True
        )
    else:
        # Create a dummy wandb object to avoid errors in logging code
        logger.info('[INFO] wandb disabled (trial mode or --no_wandb)')
        wandb = misc.BlackHole()
    
    # Trial mode header
    if args.trial:
        logger.info("=" * 60)
        logger.info("TRIAL MODE - Quick Testing")
        logger.info("=" * 60)
        logger.info(f"Log directory: {log_dir}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Max iterations: {config.train.max_iters}")
        logger.info(f"Batch size: {config.train.batch_size}")
        logger.info(f"Workers: {config.train.num_workers}")
        logger.info("=" * 60)

    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
        # heavy_only=config.data.heavy_only
    )
    train_set, val_set = subsets['train'], subsets['test']

    # Trial mode: use subset of data
    if args.trial:
        trial_train_size = min(1000, len(train_set))
        trial_val_size = min(100, len(val_set))
        train_set = torch.utils.data.Subset(train_set, range(trial_train_size))
        val_set = torch.utils.data.Subset(val_set, range(trial_val_size))
        logger.info(f'[TRIAL] Using subset - Training: {len(train_set)}, Validation: {len(val_set)}')
    else:
        logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    # print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)


    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            # Trial mode: log first batch info
            if args.trial and it == 1:
                logger.info(f'[TRIAL] First batch - protein atoms: {batch.protein_pos.shape[0]}, '
                           f'ligand atoms: {batch.ligand_pos.shape[0]}')

            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch
            )
            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )

            # Trial mode: extra debug info
            if args.trial and args.device == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(args.device) / 1e9
                mem_reserved = torch.cuda.memory_reserved(args.device) / 1e9
                logger.info(f'[TRIAL] GPU Memory: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB')

            log_dict = {'iteration': it}
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
                    log_dict[f'train/{k}'] = v.item()
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

            # Log to wandb (only if not in trial/no_wandb mode)
            if use_wandb:
                log_dict.update({
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/grad': orig_grad_norm,
                })
                wandb.log(log_dict)


    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        sum_loss_bond, sum_loss_non_bond = 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []

        # Trial mode: use fewer timesteps for faster validation
        num_val_timesteps = 5 if args.trial else 10

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate', disable=not args.trial):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                t_loss, t_loss_pos, t_loss_v = [], [], []
                for t in np.linspace(0, model.num_timesteps - 1, num_val_timesteps).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step
                    )
                    loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        # Compute AUROC (skip in trial mode for speed)
        if not args.trial:
            atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                                   feat_mode=config.data.transform.ligand_atom_mode)
            logger.info(
                '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                    it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
                )
            )
        else:
            logger.info(
                '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3' % (
                    it, avg_loss, avg_loss_pos, avg_loss_v * 1000
                )
            )

        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.flush()

        # Log to wandb (only if not in trial/no_wandb mode)
        if use_wandb:
            log_dict = {
                'iteration': it,
                'val/loss': avg_loss,
                'val/loss_pos': avg_loss_pos,
                'val/loss_v': avg_loss_v,
            }
            if not args.trial:
                log_dict['val/atom_auroc'] = atom_auroc
            wandb.log(log_dict)

        return avg_loss


    # Training loop
    import time
    start_time = time.time()

    try:
        best_loss, best_iter = None, None
        for it in range(1, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)

                    # Log best model to wandb (only if enabled)
                    if use_wandb:
                        wandb.run.summary['best_val_loss'] = best_loss
                        wandb.run.summary['best_iter'] = best_iter
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')

        # Trial mode summary
        if args.trial:
            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info("TRIAL COMPLETED")
            logger.info(f"Total time: {elapsed/60:.1f} minutes")
            logger.info(f"Best val loss: {best_loss:.6f} at iteration {best_iter}")
            logger.info(f"Checkpoints saved to: {ckpt_dir}")
            logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info('Terminating...')
    finally:
        if use_wandb:
            wandb.finish()
