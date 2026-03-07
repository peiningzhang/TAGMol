"""
Trial training script for VEDA - lightweight version for quick testing.
Features:
- No wandb logging
- No sbatch dependency
- Trial mode with extra debug logs
- Minimal checkpoint saving
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion_trial', help='Log directory for trial runs')
    parser.add_argument('--tag', type=str, default='trial', help='Tag for this trial run')
    parser.add_argument('--train_report_iter', type=int, default=50, help='Report interval (smaller for trial)')
    parser.add_argument('--max_iters', type=int, default=1000, help='Max iterations for trial (default: 1000)')
    parser.add_argument('--val_freq', type=int, default=200, help='Validation frequency (default: 200)')
    parser.add_argument('--trial', action='store_true', default=True, help='Enable trial mode with extra logs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from config')
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Override config for trial mode
    config.train.max_iters = args.max_iters
    config.train.val_freq = args.val_freq
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    config.train.num_workers = args.num_workers

    # Logging (No wandb in trial mode)
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = misc.get_logger('train_trial', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    # Trial mode header
    logger.info("=" * 60)
    logger.info("TRIAL MODE - VEDA Training Test")
    logger.info("=" * 60)
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max iterations: {args.max_iters}")
    logger.info(f"Batch size: {config.train.batch_size}")
    logger.info(f"Workers: {config.train.num_workers}")
    logger.info("=" * 60)

    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

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
    if args.trial:
        logger.info('[TRIAL] Using subset of data for quick testing')

    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    if args.trial:
        # Use small subset for trial
        train_set = torch.utils.data.Subset(train_set, range(min(1000, len(train_set))))
        val_set = torch.utils.data.Subset(val_set, range(min(100, len(val_set))))
        logger.info(f'[TRIAL] Subset - Training: {len(train_set)} Validation: {len(val_set)}')

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

    logger.info(f'Protein feature dim: {protein_featurizer.feature_dim}')
    logger.info(f'Ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    if args.trial:
        logger.info('[TRIAL] Model architecture summary:')
        logger.info(f'  - Diffusion type: {getattr(config.model, "diffusion_type", "ddpm")}')
        logger.info(f'  - Model type: {config.model.model_type}')
        logger.info(f'  - Hidden dim: {config.model.hidden_dim}')
        logger.info(f'  - Num layers: {config.model.num_layers}')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    # DFM Scheduler check
    if hasattr(config.model, 'diffusion_type') and config.model.diffusion_type == 'veda':
        from utils.misc import DFMTimeScheduler
        dfm_scheduler = DFMTimeScheduler()
        logger.info('[VEDA] DFM TimeScheduler initialized')
        if args.trial:
            # Test the scheduler
            test_t = torch.tensor([0.0, 0.5, 1.0, 10.0])
            kappa_vals = dfm_scheduler.kappa(test_t)
            dkappa_vals = dfm_scheduler.d_kappa_dt(test_t)
            logger.info(f'[TRIAL] DFM Scheduler test - t={test_t.tolist()}')
            logger.info(f'[TRIAL]   kappa(t)={kappa_vals.tolist()}')
            logger.info(f'[TRIAL]   d_kappa/dt={dkappa_vals.tolist()}')

    def train(it):
        model.train()
        optimizer.zero_grad()

        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            if args.trial and it == 1:
                logger.info(f'[TRIAL] First batch - protein atoms: {batch.protein_pos.shape[0]}, ligand atoms: {batch.ligand_pos.shape[0]}')
                logger.info(f'[TRIAL] Batch protein batch: {batch.protein_element_batch.unique().tolist()}')

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

            if args.trial:
                # Extra debug info in trial mode
                mem_allocated = torch.cuda.memory_allocated(args.device) / 1e9 if args.device == 'cuda' else 0
                mem_reserved = torch.cuda.memory_reserved(args.device) / 1e9 if args.device == 'cuda' else 0
                logger.info(f'[TRIAL] GPU Memory: Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB')

            log_dict = {'iteration': it}
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
                    log_dict[f'train/{k}'] = v.item()
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

    def validate(it):
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        all_pred_v, all_true_v = [], []

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate', disable=not args.trial):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 5).astype(int):  # Fewer timesteps for trial
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

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000
            )
        )

        if args.trial:
            atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                                   feat_mode=config.data.transform.ligand_atom_mode)
            logger.info(f'[TRIAL] Validation AUROC: {atom_auroc:.4f}')

        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.flush()

        return avg_loss

    # Training loop
    try:
        best_loss, best_iter = None, None
        start_time = time.time()

        for it in range(1, config.train.max_iters + 1):
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
                    logger.info(f'[TRIAL] Checkpoint saved: {ckpt_path}')
                else:
                    logger.info(f'[Validate] Val loss not improved. Best: {best_loss:.6f} at iter {best_iter}')

        # Trial summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("TRIAL COMPLETED")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Best val loss: {best_loss:.6f} at iteration {best_iter}")
        logger.info(f"Checkpoints saved to: {ckpt_dir}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info('Terminating trial...')
    except Exception as e:
        logger.error(f'Trial failed with error: {e}')
        import traceback
        logger.error(traceback.format_exc())
        raise
