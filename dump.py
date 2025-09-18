import argparse
import os
import os.path as osp
from typing import Optional

import numpy as np
import torch

from exp import BaseExperiment


def parse_args():
    parser = argparse.ArgumentParser(description='Dump predictions from checkpoint')
    parser.add_argument('--ex_name', '-ex', required=True, type=str, help='Experiment name under work_dirs')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint (defaults to best.ckpt under work_dirs/ex_name)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')
    parser.add_argument('--index', '-i', type=int, default=0, help='Batch index to dump')
    parser.add_argument('--data_root', type=str, default=None, help='Override data_root for dataset (e.g., single .nc file)')
    # parser.add_argument('--high_res', action='store_true', default=False, help='Use high resolution inference with INR decoder')
    parser.add_argument('--output_size', nargs=2, type=int, default=[128, 256], help='Output resolution for high-res inference [H, W]')
    parser.add_argument('--aft_seq_length', type=int, default=None, help='Number of future frames to predict')
    parser.add_argument('--train_mask', type=float, default=1.0, help='Fraction of coordinates to use (1.0 = full resolution)')
    parser.add_argument('--time_interval', type=float, default=None, help='Time interval in hours (default: 6)')
    return parser.parse_args()


@torch.no_grad()
def main():
    cli_args = parse_args()

    # Load checkpoint to get saved hyperparameters
    if cli_args.ckpt_path is None:
        # Try best.ckpt first, then last.ckpt
        best_ckpt = osp.join('work_dirs', cli_args.ex_name, 'checkpoints', 'best.ckpt')
        last_ckpt = osp.join('work_dirs', cli_args.ex_name, 'checkpoints', 'last.ckpt')
        if osp.exists(best_ckpt):
            ckpt_path = best_ckpt
        elif osp.exists(last_ckpt):
            ckpt_path = last_ckpt
        else:
            ckpt_path = cli_args.ex_name
    else:
        ckpt_path = cli_args.ckpt_path
    
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Load saved hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        saved_hparams = checkpoint['hyper_parameters']
        print(f"Loaded hyperparameters from checkpoint: {list(saved_hparams.keys())}")
        
        # Create args object from saved hyperparameters
        class Args:
            def __init__(self, hparams):
                for key, value in hparams.items():
                    setattr(self, key, value)
        
        args = Args(saved_hparams)
        
        # Set missing required fields with defaults
        if not hasattr(args, 'levels') or args.levels is None:
            args.levels = [850, 500, 1000]  # Default levels for weather data
        if not hasattr(args, 'time_interval') or args.time_interval is None:
            args.time_interval = cli_args.time_interval if cli_args.time_interval is not None else 1
        if not hasattr(args, 'coord_encoding') or args.coord_encoding is None:
            args.coord_encoding = 'hashgrid'  # Default coordinate encoding
        if not hasattr(args, 'encode_rel_coords') or args.encode_rel_coords is None:
            args.encode_rel_coords = True  # Default relative coordinates
        if not hasattr(args, 'hash_num_levels') or args.hash_num_levels is None:
            args.hash_num_levels = 16  # Default hash grid levels
        if not hasattr(args, 'hash_features_per_level') or args.hash_features_per_level is None:
            args.hash_features_per_level = 2  # Default features per level
        if not hasattr(args, 'hash_log2_hashmap_size') or args.hash_log2_hashmap_size is None:
            args.hash_log2_hashmap_size = 19  # Default hashmap size
        if not hasattr(args, 'hash_base_resolution') or args.hash_base_resolution is None:
            args.hash_base_resolution = 16  # Default base resolution
        if not hasattr(args, 'hash_per_level_scale') or args.hash_per_level_scale is None:
            args.hash_per_level_scale = 1.447  # Default per-level scale
        if not hasattr(args, 'inr_chunk') or args.inr_chunk is None:
            args.inr_chunk = 10000  # Default chunk size
        if not hasattr(args, 'train_mask') or args.train_mask is None:
            args.train_mask = 1.0  # Default full resolution
        if not hasattr(args, 'pred_beta') or args.pred_beta is None:
            args.pred_beta = 1.0  # Default prediction loss weight
            
        # Override with CLI arguments if provided
        if cli_args.data_root is not None:
            args.data_root = cli_args.data_root
        if cli_args.device is not None:
            args.device = cli_args.device
        # if cli_args.high_res:
        #     args.high_res = cli_args.high_res
        if cli_args.output_size is not None:
            args.output_size = cli_args.output_size
        if cli_args.train_mask is not None:
            args.train_mask = cli_args.train_mask
        if cli_args.time_interval is not None:
            args.time_interval = cli_args.time_interval
            
        # Set CLI-specific attributes
        # args.high_res = cli_args.high_res
        args.output_size = cli_args.output_size
        args.aft_seq_length = cli_args.aft_seq_length if cli_args.aft_seq_length is not None else args.aft_seq_length
        args.train_mask = cli_args.train_mask
        args.time_interval = cli_args.time_interval if cli_args.time_interval is not None else args.time_interval
        
        print(f"Using time_interval: {args.time_interval}")
            
    else:
        print("Error: No hyperparameters found in checkpoint!")
        return

    # Force time ranges to match single-year NetCDF if needed
    args.train_time = ['2015', '2015']
    args.val_time = ['2016', '2016']
    args.test_time = ['2017', '2017']
    if "0p25" in args.data_root:
        args.train_time = ['2017-01-01T00', '2017-01-10T06']
        args.val_time = ['2017-01-01T00', '2017-01-10T06']
        args.test_time = ['2017-01-01T00', '2017-01-10T06']
    
    # Set test mode
    args.test = True

    exp = BaseExperiment(args)

    # Device
    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    exp.method.to(device)

    # Load model state (use the same checkpoint path as before)
    try:
        state = torch.load(ckpt_path, map_location=device)
        exp.method.load_state_dict(state['state_dict'], strict=False)
        print("Model loaded successfully (some layers may be missing due to architecture changes)")
    except Exception as e:
        print(f"Warning: Could not load checkpoint due to architecture mismatch: {e}")
        print("Using randomly initialized model for testing")

    # One batch from test loader
    with torch.no_grad():
        for loader, split in [(exp.data.test_loader, 'test'), (exp.data.train_loader, 'train')]:
            bx, by = next(iter(loader))
            bx = bx.to(device, non_blocking=True)

            # # Forward: use model's internal structure for proper inference
            # if args.high_res and hasattr(exp.method.model, 'inr_dec') and exp.method.model.inr_dec is not None:
            #     # High resolution inference using INR decoder
            #     print(f"High resolution inference: {args.output_size}")
                
            #     with torch.no_grad():
            #         # Get the model's internal processing
            #         B, T, C, H, W = bx.shape
            #         x = bx.view(B*T, C, H, W)
                    
            #         # Encoder
            #         embed, skip = exp.method.model.enc(x)
            #         _, C_, H_, W_ = embed.shape
            #         C_small = embed.shape[1]
            #         z = embed.view(B, T, C_small, H_, W_)
                    
            #         # Align runtime T with construction-time T
            #         if T != exp.method.model.init_T:
            #             if T < exp.method.model.init_T:
            #                 pad_n = exp.method.model.init_T - T
            #                 pad_tail = z[:, -1:].repeat(1, pad_n, 1, 1, 1)
            #                 z = torch.cat([z, pad_tail], dim=1)
            #             else:
            #                 z = z[:, :exp.method.model.init_T]
            #             T = exp.method.model.init_T
                    
            #         # Hidden processing
            #         hid = exp.method.model.hid(z)
            #         hid = hid.reshape(B*T, C_small, H_, W_)
                    
            #         # Neural ODE prediction if enabled
            #         if exp.method.model.use_neuralode:
            #             pred_steps = int(args.aft_seq_length)
            #             z_btchw = hid.view(B, T, C_small, H_, W_)
            #             z0 = z_btchw[:, -1]
            #             z_future = exp.method.model.latent_ode(z0, steps=pred_steps)
            #             dec_seq = z_future.reshape(B*pred_steps, C_small, H_, W_)
            #             T_out = pred_steps
            #         else:
            #             dec_seq = hid
            #             T_out = T
                    
            #         # INR decoder with custom output size
            #         out = exp.method.model.inr_dec(
            #             dec_seq,
            #             output_size=args.output_size,
            #             bsize=args.inr_chunk,
            #             train_mask=1.0  # Full resolution for inference
            #         )
                    
            #         if isinstance(out, dict):
            #             preds = out['pred'].reshape(B, T_out, -1, args.output_size[0], args.output_size[1])
            #         else:
            #             preds = out.reshape(B, T_out, -1, args.output_size[0], args.output_size[1])
            # else:
            
            
            # Forward with optional output size and aft_seq_length
            print(f"Inference: output_size={args.output_size}, aft_seq_length={args.aft_seq_length}")
            out = exp.method.model(bx, output_size=args.output_size, aft_seq_length=args.aft_seq_length)
            
            preds = out[0] if isinstance(out, tuple) else out

            # Save
            save_dir = osp.join('work_dirs', args.ex_name, 'saved')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save inputs
            np.save(osp.join(save_dir, split+'_inputs.npy'), bx.detach().cpu().numpy())
            
            # Save ground-truth future frames for prediction visualization
            np.save(osp.join(save_dir, split+'_trues.npy'), by.detach().cpu().numpy())
            
            # Save predictions
            if isinstance(preds, dict):
                # Handle masked output
                pred_array = preds['pred'].detach().cpu().numpy()
                np.save(osp.join(save_dir, split+'_preds.npy'), pred_array)
                if 'mask' in preds:
                    np.save(osp.join(save_dir, split+'_mask.npy'), preds['mask'].detach().cpu().numpy())
            else:
                np.save(osp.join(save_dir, split+'_preds.npy'), preds.detach().cpu().numpy())
            
            print(f'Saved to {save_dir}: {list(sorted(os.listdir(save_dir)))}')
            print(f'Input shape: {bx.shape}, Prediction shape: {preds.shape if not isinstance(preds, dict) else preds["pred"].shape}')
        

if __name__ == '__main__':
    main()


