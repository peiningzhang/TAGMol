import argparse
import os
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
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
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
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument(
        '--reuse_logdir',
        type=str,
        default=None,
        help='Use this existing directory as log_dir (checkpoints, log.txt, tensorboard). '
        'Use when resuming so new ckpts append to the same run folder.',
    )
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    # Trial mode options
    parser.add_argument('--trial', action='store_true', help='Enable trial mode for quick testing')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--max_iters', type=int, default=None, help='Override max iterations')
    parser.add_argument('--val_freq', type=int, default=None, help='Override validation frequency')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='Override num workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint (path to checkpoint file)')
    parser.add_argument('--wandb_id', type=str, default=None, help='Wandb run ID to resume')
    parser.add_argument('--wandb_resume', type=str, default=None, 
                       choices=['allow', 'must', 'never'], 
                       help='Wandb resume mode: allow (resume if exists), must (require resume), never (always new)')
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
    use_wandb = not args.no_wandb and not args.trial

    if args.trial:
        # Trial mode: no log dir, use console logger only
        logger = misc.get_logger('train', log_dir=None)
        writer = misc.BlackHole()  # Dummy writer
        ckpt_dir = None
        log_dir = None
    else:
        if args.reuse_logdir:
            log_dir = os.path.abspath(os.path.expanduser(args.reuse_logdir))
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        vis_dir = os.path.join(log_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        logger = misc.get_logger('train', log_dir)
        if args.reuse_logdir:
            logger.info(f'Reusing log directory: {log_dir}')
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    # Initialize wandb (skip if --no_wandb or --trial)
    if use_wandb:
        wandb_init_kwargs = {
            'project': "tagmol",
            'name': f"{config_name}_{args.tag}" if args.tag else config_name,
            'config': config,
            'dir': log_dir,
            'save_code': True
        }
        
        # Handle wandb resume logic
        if args.wandb_id:
            # Resume specific run by ID
            wandb_init_kwargs['id'] = args.wandb_id
            wandb_init_kwargs['resume'] = 'must'  # Must resume this specific run
            logger.info(f'Resuming wandb run with ID: {args.wandb_id}')
        elif args.wandb_resume:
            # Use resume mode (allow/must/never)
            wandb_init_kwargs['resume'] = args.wandb_resume
            if args.wandb_resume == 'allow':
                logger.info('Wandb resume mode: allow (will resume if run exists)')
            elif args.wandb_resume == 'must':
                logger.info('Wandb resume mode: must (will fail if run does not exist)')
            elif args.wandb_resume == 'never':
                logger.info('Wandb resume mode: never (always create new run)')
        else:
            # Default: create new run
            logger.info('Creating new wandb run')
        
        wandb.init(**wandb_init_kwargs)
        
        # Log wandb run ID for future reference
        logger.info(f'Wandb run ID: {wandb.run.id}')
        logger.info(f'Wandb run URL: {wandb.run.url}')
    else:
        # Create a dummy wandb object to avoid errors in logging code
        if args.trial:
            logger.info('[INFO] Trial mode: wandb and file logging disabled')
        elif args.no_wandb:
            logger.info('[INFO] wandb disabled (--no_wandb)')
        wandb = misc.BlackHole()

    # Trial mode header
    if args.trial:
        logger.info("=" * 60)
        logger.info("TRIAL MODE - Quick Testing")
        logger.info("=" * 60)
        logger.info(f"Device: {args.device}")
        logger.info(f"Max iterations: {config.train.max_iters}")
        logger.info(f"Batch size: {config.train.batch_size}")
        logger.info(f"Workers: {config.train.num_workers}")
        logger.info("=" * 60)

    logger.info(args)
    logger.info(config)

    if not args.trial:
        config_dst = os.path.join(log_dir, os.path.basename(args.config))
        try:
            shutil.copyfile(args.config, config_dst)
        except shutil.SameFileError:
            pass  # config already lives in log_dir (e.g. resume with training.yml in reuse_logdir)
        models_snap = os.path.join(log_dir, 'models')
        if not os.path.isdir(models_snap):
            shutil.copytree('./models', models_snap)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    condition_bins_path = getattr(config.data.transform, 'condition_bins_path', None)
    if condition_bins_path:
        transform_list.append(trans.FeaturizeConditionBins(spec_path=condition_bins_path))
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
    # Optimize DataLoader for better GPU utilization
    # pin_memory=True: Faster CPU->GPU transfer
    # persistent_workers=True: Keep workers alive between epochs
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True if args.device == 'cuda' else False,  # Enable pin_memory for CUDA
        persistent_workers=True if config.train.num_workers > 0 else False,
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
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
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
        
        # Load model state
        model.load_state_dict(ckpt['model'])
        logger.info('Loaded model state')
        
        # Load optimizer state
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info('Loaded optimizer state')
        
        # Load scheduler state
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            logger.info('Loaded scheduler state')
        
        # Load training state
        if 'iteration' in ckpt:
            start_iter = ckpt['iteration'] + 1
            logger.info(f'Resuming from iteration: {start_iter}')
        
        if 'best_loss' in ckpt:
            best_loss = ckpt['best_loss']
            logger.info(f'Previous best loss: {best_loss:.6f}')
        
        if 'best_iter' in ckpt:
            best_iter = ckpt['best_iter']
            logger.info(f'Previous best iteration: {best_iter}')
        
        # Use config from checkpoint if available (for compatibility)
        if 'config' in ckpt and not args.trial:
            logger.info('Note: Using config from command line, not checkpoint')
        
        logger.info('Checkpoint loaded successfully!')


    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            # Trial mode: log first batch info and verify kappa-t mapping
            if args.trial and it == 1:
                logger.info(f'[TRIAL] First batch - protein atoms: {batch.protein_pos.shape[0]}, '
                           f'ligand atoms: {batch.ligand_pos.shape[0]}')

            results = model.get_diffusion_loss(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
                ligand_bond_index=batch.ligand_bond_index,
                ligand_bond_type=batch.ligand_bond_type,
                ligand_bond_type_batch=batch.ligand_bond_type_batch,
                vina_bin=getattr(batch, 'vina_bin', None),
                qed_bin=getattr(batch, 'qed_bin', None),
                sa_bin=getattr(batch, 'sa_bin', None),
            )
            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            loss_bond = results.get('loss_bond', torch.tensor(0.))
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | bond %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, loss_bond, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )

            # Trial mode: extra debug info
            if args.trial and args.device == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(args.device) / 1e9
                mem_reserved = torch.cuda.memory_reserved(args.device) / 1e9

            log_dict = {'iteration': it}
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    if not args.trial:
                        writer.add_scalar(f'train/{k}', v, it)
                    log_dict[f'train/{k}'] = v.item()
            if not args.trial:
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
                        time_step=time_step,
                        ligand_bond_index=batch.ligand_bond_index,
                        ligand_bond_type=batch.ligand_bond_type,
                        ligand_bond_type_batch=batch.ligand_bond_type_batch,
                        vina_bin=getattr(batch, 'vina_bin', None),
                        qed_bin=getattr(batch, 'qed_bin', None),
                        sa_bin=getattr(batch, 'sa_bin', None),
                    )
                    loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
                    loss_bond = results.get('loss_bond', torch.tensor(0.))

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_bond += float(loss_bond) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_bond = sum_loss_bond / sum_n

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        # Compute AUROC
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)
        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss bond %.6f | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_bond, atom_auroc
            )
        )

        if not args.trial:
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
                'val/atom_auroc': atom_auroc,
            }
            wandb.log(log_dict)

        return avg_loss


    # Quick-eval (sampling + full metrics) every N steps for wandb
    quick_eval_freq = getattr(config.train, 'quick_eval_freq', 10000)
    quick_eval_num_proteins = getattr(config.train, 'quick_eval_num_proteins', 10)
    quick_eval_num_ligands = getattr(config.train, 'quick_eval_num_ligands_per_protein', 10)

    # Training loop
    import time
    start_time = time.time()
    if not args.trial:
        yml_path = os.path.join(log_dir, 'training.yml')
        if os.path.exists(yml_path):
            os.system(f"cp {yml_path} {yml_path.replace('training.yml', 'sampling.yml')}")
    try:
        # best_loss and best_iter are initialized above (from checkpoint if resuming)
        for it in range(start_iter, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)
            
            # Save last.pt at every iteration (skip in trial mode)
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
                logger.info(f'[Checkpoint] Saved last checkpoint to: {last_ckpt_path} (iter {it})')
            # Quick eval: sample + full metrics every quick_eval_freq steps, log to wandb
            if (it % quick_eval_freq == 0) or (it == config.train.max_iters):
                eval_scales = [0.0, 3.0] if getattr(config.model, 'use_condition', False) else [0.0]
                for cfg_scale in eval_scales:
                    tmp_dir = tempfile.mkdtemp(prefix=f'train_quick_eval_cfg{cfg_scale}_', dir=log_dir)
                    try:
                        model.eval()
                        n_pocket = min(quick_eval_num_proteins, len(val_set))
                        for data_id in range(n_pocket):
                            data = val_set[data_id]
                            
                            # For de novo sampling, force condition bins to the highest tier
                            if getattr(config.model, 'use_condition', False):
                                num_bins = getattr(config.model, 'condition_bins', 5)
                                best_bin = num_bins - 1
                                data.vina_bin = torch.tensor(best_bin, dtype=torch.long)
                                data.qed_bin = torch.tensor(best_bin, dtype=torch.long)
                                data.sa_bin = torch.tensor(best_bin, dtype=torch.long)
                                
                            with torch.no_grad():
                                pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, pred_pos0_traj, time_list = sample_diffusion_ligand(
                                    model, data, quick_eval_num_ligands,
                                    batch_size=min(quick_eval_num_ligands, 10),
                                    device=args.device,
                                    num_steps=config.model.num_diffusion_timesteps,
                                    pos_only=False,
                                    center_pos_mode=config.model.center_pos_mode,
                                    sample_num_atoms='prior',
                                    cfg_scale=cfg_scale,
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
                        test_protein_root = config.data.test_path
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
                        
                        prefix = 'eval' if cfg_scale == 0.0 else f'eval_cfg{int(cfg_scale)}'
                        eval_log = {f'{prefix}/{k}': v for k, v in metrics.items() if v is not None}
                        if eval_log and use_wandb:
                            wandb.log(eval_log)
                        
                        log_parts = ['%s=%.4f' % (k, v) for k, v in list(metrics.items())[:8] if v is not None]
                        scale_str = f' (cfg={cfg_scale})' if cfg_scale != 0.0 else ''
                        logger.info(f'[QuickEval] Iter {it}{scale_str} | {" ".join(log_parts)}')
                        
                        # Vina Score / Vina Min (Mean, Median)
                        vs_mean, vs_med = metrics.get('Vina_score_mean'), metrics.get('Vina_score_med')
                        vm_mean, vm_med = metrics.get('Vina_min_mean'), metrics.get('Vina_min_med')
                        if vs_mean is not None and vs_med is not None:
                            logger.info(f'[QuickEval]{scale_str} Vina Score:  Mean: {vs_mean:.3f}  Median: {vs_med:.3f}')
                        if vm_mean is not None and vm_med is not None:
                            logger.info(f'[QuickEval]{scale_str} Vina Min  :  Mean: {vm_mean:.3f}  Median: {vm_med:.3f}')
                            
                    except Exception as e:
                        logger.error(f'Error during QuickEval (cfg={cfg_scale}): {e}')
                    finally:
                        model.train()
                        if os.path.isdir(tmp_dir):
                            shutil.rmtree(tmp_dir, ignore_errors=True)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it

                    # Save checkpoint (skip in trial mode)
                    if not args.trial:
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
            logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info('Terminating...')
    finally:
        # Always save last.pt even if training is interrupted
        if not args.trial and 'it' in locals():
            try:
                last_ckpt_path = os.path.join(ckpt_dir, 'last.pt')
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'best_loss': best_loss if 'best_loss' in locals() else None,
                    'best_iter': best_iter if 'best_iter' in locals() else None,
                }, last_ckpt_path)
                logger.info(f'[Checkpoint] Saved last checkpoint to: {last_ckpt_path}')
            except Exception as e:
                logger.warning(f'Failed to save last checkpoint: {e}')
        
        if use_wandb:
            wandb.finish()
