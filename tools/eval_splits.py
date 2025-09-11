import argparse
import os
import os.path as osp
from typing import List

import numpy as np
import torch

from openstl.methods import method_maps
from openstl.utils import load_config, update_config, default_parser, get_dataset, check_dir, print_log


def parse_args():
    parser = argparse.ArgumentParser(description='Dump predictions for specific splits using an existing checkpoint')
    parser.add_argument('--config_file', '-c', type=str, required=True, help='Path to the default config file')
    parser.add_argument('--dataname', '-d', type=str, required=True, help='Dataset name, e.g., mmnist')
    parser.add_argument('--method', '-m', type=str, default='SimVP', help='Method name, e.g., SimVP')
    parser.add_argument('--res_dir', type=str, default='work_dirs', help='Base results directory')
    parser.add_argument('--ex_name', '-ex', type=str, required=True, help='Experiment name under res_dir')
    parser.add_argument('--splits', type=str, default='train,val', help='Comma-separated splits to dump: train,val,test')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    return parser.parse_args()


def _compose_save_dir(res_dir: str, ex_name: str) -> str:
    base_dir = res_dir if res_dir is not None else 'work_dirs'
    save_dir = osp.join(base_dir, ex_name if not ex_name.startswith(base_dir) else ex_name.split(base_dir + '/')[-1])
    return save_dir


@torch.no_grad()
def _dump_for_split(split_name: str, model: torch.nn.Module, device: torch.device, loader, out_dir: str):
    model.eval()
    all_inputs: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    total_batches = len(loader)
    print_log(f"[{split_name}] num_batches={total_batches}")
    for idx, batch in enumerate(loader):
        batch_x, batch_y = batch
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        pred_y = model(batch_x, batch_y)
        all_inputs.append(batch_x.detach().cpu().numpy())
        all_trues.append(batch_y.detach().cpu().numpy())
        all_preds.append(pred_y.detach().cpu().numpy())
        if (idx + 1) % 50 == 0 or (idx + 1) == total_batches:
            print_log(f"[{split_name}] processed {idx + 1}/{total_batches}")

    inputs = np.concatenate(all_inputs, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    save_dir = check_dir(out_dir)
    np.save(osp.join(save_dir, 'inputs.npy'), inputs)
    np.save(osp.join(save_dir, 'trues.npy'), trues)
    np.save(osp.join(save_dir, 'preds.npy'), preds)


def main():
    args = parse_args()

    # Build config similarly to tools/test.py
    loaded_cfg = load_config(args.config_file)
    cfg = {
        'dataname': args.dataname,
        'method': args.method,
    }
    cfg = update_config(cfg, loaded_cfg, exclude_keys=['method', 'val_batch_size'])
    defaults = default_parser()
    for k, v in defaults.items():
        if cfg.get(k, None) is None:
            cfg[k] = v

    # Compose save_dir and load dataloaders
    save_dir = _compose_save_dir(args.res_dir, args.ex_name)
    train_loader, vali_loader, test_loader = get_dataset(args.dataname, cfg)
    print_log(f"dataloaders: train={len(train_loader)}, val={len(vali_loader)}, test={len(test_loader)}")

    # Build method and load checkpoint
    method_name = args.method.lower()
    steps_per_epoch = len(train_loader)
    method = method_maps[method_name](steps_per_epoch=steps_per_epoch,
                                     test_mean=test_loader.dataset.mean,
                                     test_std=test_loader.dataset.std,
                                     save_dir=save_dir,
                                     **cfg)

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    method.to(device)

    ckpt_path = osp.join(save_dir, 'checkpoints', 'best.ckpt')
    print_log(f"loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    method.load_state_dict(state['state_dict'])

    # Determine splits to run
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    for split in splits:
        print_log(f"start split: {split}")
        if split == 'train':
            out_dir = osp.join(save_dir, 'saved_train')
            _dump_for_split('train', method, device, train_loader, out_dir)
        elif split in ('val', 'vali', 'valid', 'validation'):
            out_dir = osp.join(save_dir, 'saved_val')
            _dump_for_split('val', method, device, vali_loader, out_dir)
        elif split == 'test':
            out_dir = osp.join(save_dir, 'saved_test')
            _dump_for_split('test', method, device, test_loader, out_dir)
        else:
            raise ValueError(f'Unknown split: {split}')
        print_log(f"saved split '{split}' to {out_dir}")


if __name__ == '__main__':
    main()


