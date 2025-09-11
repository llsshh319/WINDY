import argparse
import os
import os.path as osp
from typing import Optional

import numpy as np
import torch

from api import BaseExperiment
from utils import load_config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Dump a batch of predictions from a checkpoint')
    parser.add_argument('--config_file', '-c', required=True, type=str, help='Path to config file')
    parser.add_argument('--dataname', '-d', required=True, type=str, help='Dataset name (e.g., weather_t2m_1_40625)')
    parser.add_argument('--ex_name', '-ex', required=True, type=str, help='Experiment name under work_dirs')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint (defaults to best.ckpt under work_dirs/ex_name)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    parser.add_argument('--index', '-i', type=int, default=0, help='Batch index to dump')
    parser.add_argument('--data_root', type=str, default=None, help='Override data_root for dataset (e.g., single .nc file)')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    # Build config similar to eval_splits/tools/train
    loaded_cfg = load_config(args.config_file)
    cfg = {
        'dataname': args.dataname,
        'method': 'simvp',
    }
    cfg = update_config(cfg, loaded_cfg, exclude_keys=['method', 'val_batch_size'])
    if args.data_root is not None:
        cfg['data_root'] = args.data_root
    # Force time ranges to match single-year NetCDF if needed
    cfg['train_time'] = ['2010', '2010']
    cfg['val_time'] = ['2010', '2010']
    cfg['test_time'] = ['2010', '2010']
    # Argparser defaults are not available here; keep cfg keys as provided/loaded
    cfg['ex_name'] = args.ex_name
    cfg['test'] = True
    # Convert to args-like object
    class A: pass
    a = A()
    for k, v in cfg.items():
        setattr(a, k, v)

    exp = BaseExperiment(a)

    # Device
    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    exp.method.to(device)

    # Checkpoint
    if args.ckpt_path is None:
        ckpt_path = osp.join('work_dirs', args.ex_name, 'checkpoints', 'best.ckpt')
    else:
        ckpt_path = args.ckpt_path
    state = torch.load(ckpt_path, map_location=device)
    exp.method.load_state_dict(state['state_dict'])

    # One batch from test loader
    for loader, split in [(exp.data.test_loader, 'test'), (exp.data.train_loader, 'train')]:

        # if len(loader) == 0:
        #     loader = exp.data.train_loader
        bx, by = next(iter(loader))
        bx = bx.to(device, non_blocking=True)

        # Forward: use method forward to respect ODE multi-step prediction
        out = exp.method(bx)
        preds = out[0] if isinstance(out, tuple) else out

        # Save
        save_dir = osp.join('work_dirs', args.ex_name, 'saved')
        os.makedirs(save_dir, exist_ok=True)
        np.save(osp.join(save_dir, split+'_inputs.npy'), bx.detach().cpu().numpy())
        # Save ground-truth future frames for prediction visualization
        np.save(osp.join(save_dir, split+'_trues.npy'), by.detach().cpu().numpy())
        np.save(osp.join(save_dir, split+'_preds.npy'), preds.detach().cpu().numpy())
        print('Saved to', save_dir, list(sorted(os.listdir(save_dir))))
    

if __name__ == '__main__':
    main()


