import argparse
import os
import glob
import shutil
import tempfile

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
from models.molopt_guide_model import DockGuideNet3D
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_guided_diffusion import sample_guided_diffusion_ligand
from scripts.evaluate_diffusion import run_evaluation


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
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--trial', action='store_true', help='Trial mode for quick testing')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--max_iters', type=int, default=None, help='Override max iterations')
    parser.add_argument('--val_freq', type=int, default=None, help='Override validation frequency')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='Override num workers')
    parser.add_argument('--wandb_id', type=str, default=None, help='Wandb run ID to resume')
    parser.add_argument('--wandb_resume', type=str, default=None,
                       choices=['allow', 'must', 'never'], help='Wandb resume mode')
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Trial mode: override config (aligned with train_diffusion.py)
    if args.trial:
        if args.max_iters is not None:
            config.train.max_iters = args.max_iters
        if args.val_freq is not None:
            config.train.val_freq = args.val_freq
        if args.batch_size is not None:
            config.train.batch_size = args.batch_size
        if args.num_workers is not None:
            config.train.num_workers = args.num_workers
        if args.train_report_iter == 200:
            args.train_report_iter = 50

    use_wandb = not args.no_wandb and not args.trial

    if args.trial:
        logger = misc.get_logger('train', log_dir=None)
        writer = misc.BlackHole()
        ckpt_dir = None
        log_dir = None
    else:
        log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        vis_dir = os.path.join(log_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        logger = misc.get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    if use_wandb:
        wandb_init_kwargs = {
            'project': "tagmol-guide",
            'name': f"{config_name}_{args.tag}" if args.tag else config_name,
            'config': config,
            'dir': log_dir,
            'save_code': True
        }
        if args.wandb_id:
            wandb_init_kwargs['id'] = args.wandb_id
            wandb_init_kwargs['resume'] = 'must'
            logger.info(f'Resuming wandb run with ID: {args.wandb_id}')
        elif args.wandb_resume:
            wandb_init_kwargs['resume'] = args.wandb_resume
        else:
            logger.info('Creating new wandb run')
        wandb.init(**wandb_init_kwargs)
        logger.info(f'Wandb run ID: {wandb.run.id}')
    else:
        wandb = misc.BlackHole()
        if args.trial:
            logger.info('[INFO] Trial mode: wandb and file logging disabled')
        elif args.no_wandb:
            logger.info('[INFO] wandb disabled (--no_wandb)')

    if args.trial:
        logger.info("=" * 60)
        logger.info("TRIAL MODE - Quick Testing")
        logger.info("=" * 60)
        logger.info(f"Max iterations: {config.train.max_iters}")
        logger.info(f"Batch size: {config.train.batch_size}")

    logger.info(args)
    logger.info(config)
    if not args.trial:
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
    get_dataset_kwargs = {'config': config.data, 'transform': transform}
    if hasattr(config.data, 'index_path') and config.data.index_path:
        get_dataset_kwargs['index_path'] = config.data.index_path
    dataset, subsets = get_dataset(**get_dataset_kwargs)
    train_set, val_set = subsets['train'], subsets['test']

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
    model = DockGuideNet3D(
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

    # Resume from checkpoint if specified
    start_iter = 1
    best_loss = None
    best_iter = None
    if args.resume is not None:
        logger.info(f'Resuming training from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        logger.info('Loaded model state')
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'iteration' in ckpt:
            start_iter = ckpt['iteration'] + 1
            logger.info(f'Resuming from iteration: {start_iter}')
        if 'best_loss' in ckpt:
            best_loss = ckpt['best_loss']
        if 'best_iter' in ckpt:
            best_iter = ckpt['best_iter']
        logger.info('Checkpoint loaded successfully')

    # Load score model for quick_eval (optional)
    score_model = None
    score_model_ckpt = getattr(config.train, 'score_model_checkpoint', None)
    if score_model_ckpt and os.path.isfile(score_model_ckpt):
        logger.info(f'Loading score model for quick_eval: {score_model_ckpt}')
        ckpt = torch.load(score_model_ckpt, map_location=args.device)
        score_cfg = ckpt.get('config', config)
        score_model = ScorePosNet3D(
            score_cfg.model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to(args.device)
        score_model.load_state_dict(ckpt['model'])
        score_model.eval()
        logger.info('Score model loaded for quick_eval')
    elif score_model_ckpt:
        logger.warning(f'score_model_checkpoint not found: {score_model_ckpt}, quick_eval disabled')

    target_mean = config.train.get("target_mean", None)
    target_std = config.train.get("target_std", None)
    normalize_target = config.train.get("normalize_target", False) and target_mean is not None and target_std is not None
    if normalize_target:
        logger.info(f'Target normalization: mean={target_mean}, std={target_std}')

    def train(it):
        model.train()
        optimizer.zero_grad()
        avg_loss = 0
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)
            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise
            dock = batch[config.train.get("target", "vina_dock")]
            if normalize_target:
                dock = (dock - target_mean) / (target_std + 1e-8)
            loss = model.get_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
                dock=dock
            )
            loss = loss / config.train.n_acc_batch
            avg_loss += loss
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f | Lr: %.6f | Grad Norm: %.6f' % (
                    it, avg_loss, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            
            writer.add_scalar('train/loss', avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()
            
            # Log to wandb
            wandb.log({
                'iteration': it,
                'train/loss': avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss,
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/grad': orig_grad_norm,
            })


    def validate(it):
        # fix time steps
        sum_loss, sum_n = 0, 0
        all_preds = []
        all_gts = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    dock = batch[config.train.get("target", "vina_dock")]
                    if normalize_target:
                        dock = (dock - target_mean) / (target_std + 1e-8)
                    loss = model.get_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step,
                        dock=dock
                    )

                    sum_loss += float(loss) * batch_size
                    sum_n += batch_size

        avg_loss = sum_loss / sum_n
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f' % (
                it, avg_loss 
            )
        )
        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        
        # Log to wandb
        wandb.log({
            'iteration': it,
            'val/loss': avg_loss,
        })
        
        return avg_loss

    quick_eval_freq = getattr(config.train, 'quick_eval_freq', 10000)
    quick_eval_num_proteins = getattr(config.train, 'quick_eval_num_proteins', 10)
    quick_eval_num_ligands = getattr(config.train, 'quick_eval_num_ligands_per_protein', 10)

    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)

            if (not args.trial) and ((it % quick_eval_freq == 0) or (it == config.train.max_iters)):
                last_ckpt_path = os.path.join(ckpt_dir, 'last.pt')
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'best_loss': best_loss,
                    'best_iter': best_iter,
                }, last_ckpt_path)
                logger.info(f'[Checkpoint] Saved last checkpoint (iter {it})')

            if ((it % quick_eval_freq == 0) or (it == config.train.max_iters)) and score_model is not None:
                tmp_base = log_dir if log_dir else tempfile.gettempdir()
                tmp_dir = tempfile.mkdtemp(prefix='train_quick_eval_', dir=tmp_base)
                try:
                    model.eval()
                    n_pocket = min(quick_eval_num_proteins, len(val_set))
                    for data_id in range(n_pocket):
                        data = val_set[data_id]
                        with torch.no_grad():
                            pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, pred_pos0_traj, time_list = sample_guided_diffusion_ligand(
                                score_model, model, data, quick_eval_num_ligands,
                                batch_size=min(quick_eval_num_ligands, 10),
                                device=args.device,
                                num_steps=config.model.num_diffusion_timesteps,
                                pos_only=False,
                                center_pos_mode=config.model.center_pos_mode,
                                sample_num_atoms='prior',
                                gradient_scale_cord=config.get("sample", {}).get("gradient_scale_cord", 1.0),
                                gradient_scale_categ=config.get("sample", {}).get("gradient_scale_categ", -10.0),
                            )
                        result = {
                            'data': data,
                            'pred_ligand_pos': pred_pos,
                            'pred_ligand_v': pred_v,
                            'pred_ligand_pos_traj': pred_pos_traj,
                            'pred_ligand_v_traj': pred_v_traj,
                            'pred_ligand_pos0_traj': pred_pos0_traj,
                            'pred_ligand_v0_traj': pred_v0_traj,
                            'time': time_list,
                        }
                        torch.save(result, os.path.join(tmp_dir, f'result_{data_id}.pt'))
                    quick_eval_docking = getattr(config.train, 'quick_eval_docking_mode', 'vina_score')
                    test_protein_root = getattr(config.data, 'test_path', './data/test_set')
                    metrics, _ = run_evaluation(
                        tmp_dir,
                        eval_step=-1,
                        eval_num_examples=n_pocket,
                        docking_mode=quick_eval_docking,
                        protein_root=test_protein_root,
                        atom_enc_mode=config.data.transform.ligand_atom_mode,
                        verbose=False,
                        save=False,
                    )
                    eval_log = {f'eval/{k}': v for k, v in metrics.items() if v is not None}
                    if eval_log and use_wandb:
                        wandb.log(eval_log)
                    log_parts = ['%s=%.4f' % (k, v) for k, v in list(metrics.items())[:8] if v is not None]
                    logger.info('[QuickEval] Iter %d | %s' % (it, ' '.join(log_parts)))
                except Exception as e:
                    import traceback
                    logger.warning('[QuickEval] Error: %s\n%s' % (e, traceback.format_exc()))
                finally:
                    model.train()
                    if os.path.isdir(tmp_dir):
                        shutil.rmtree(tmp_dir, ignore_errors=True)

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    if ckpt_dir is not None:
                        ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it,
                            'best_loss': best_loss,
                            'best_iter': best_iter,
                        }, ckpt_path)
                    if use_wandb:
                        wandb.run.summary['best_val_loss'] = best_loss
                        wandb.run.summary['best_iter'] = best_iter
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
    finally:
        if use_wandb:
            wandb.finish()
