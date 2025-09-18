import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        description='OpenSTL train/test a model')
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='Whether to use distributed training (DDP)')
    parser.add_argument('--res_dir', default='work_dirs', type=str)
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Whether to use Native AMP for mixed precision training (PyTorch=>1.6.0)')
    parser.add_argument('--torchscript', action='store_true', default=False,
                        help='Whether to use torchscripted model')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to measure inference speed (FPS)')
    parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='whether to set deterministic options for CUDNN backend (reproducable)')
    parser.add_argument('--metrics', nargs='+', default=['mse', 'mae', 'rmse'], type=str,
                        help='Metrics to evaluate model performance (default: mse mae rmse)')

    # dataset parameters
    parser.add_argument('--batch_size', '-b', default=4, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', '-vb', default=4, type=int, help='Validation batch size')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--data_root', default='/hub_data3/seunghun/2m_temperature')
    parser.add_argument('--dataname', '-d', default='weather_t2m_1_40625', type=str,
                        choices=['weather', 'weather_t2m_5_625', 'weather_mv_4_28_s6_5_625', 'weather_mv_4_4_s6_5_625',
                                'weather_r_5_625', 'weather_uv10_5_625', 'weather_tcc_5_625', 'weather_t2m_1_40625',
                                'weather_r_1_40625', 'weather_uv10_1_40625', 'weather_tcc_1_40625'],
                        help='Dataset name (default: "weather_t2m_1_40625")')

    parser.add_argument('--use_augment', action='store_true', default=False,
                        help='Whether to use image augmentations for training')
    parser.add_argument('--use_prefetcher', action='store_true', default=False,
                        help='Whether to use prefetcher for faster data loading')
    parser.add_argument('--drop_last', action='store_true', default=False,
                        help='Whether to drop the last batch in the val data loading')

    parser.add_argument('--pre_seq_length', default=1, type=int, help='Sequence length before prediction')
    parser.add_argument('--aft_seq_length', default=12, type=int, help='Sequence length after prediction')
    parser.add_argument('--time_interval', type=int, default=1,
                        help='Time interval (in frames) between indices for in/out sequences')
    parser.add_argument('--idx_in', nargs='+', type=int, default=None, help='Indices for input sequence')
    parser.add_argument('--idx_out', nargs='+', type=int, default=None, help='Indices for output sequence')
                        
    parser.add_argument('--train_time', nargs=2, type=str, default=['1979','2015'],
                        help='Train time range [start end] (e.g., 2010 2010)')
    parser.add_argument('--val_time', nargs=2, type=str, default=['2016','2016'],
                        help='Validation time range [start end]')
    parser.add_argument('--test_time', nargs=2, type=str, default=['2017','2018'],
                        help='Test time range [start end]')
    parser.add_argument('--data_name', type=str, default='t2m',
                        help='WeatherBench variable short name (e.g., t2m, r, uv10, mv)')
    parser.add_argument('--levels', nargs='+', type=int, default=[50],
                        help='Pressure levels to use (if applicable)')

    # method parameters
    parser.add_argument('--method', '-m', default='SimVP', type=str,
                        choices=['ConvLSTM', 'convlstm', 'E3DLSTM', 'e3dlstm', 'MAU', 'mau', 'MIM', 'mim', 
                                'PhyDNet', 'phydnet', 'PredRNN', 'predrnn', 'PredRNNpp',  'predrnnpp', 
                                'PredRNNv2', 'predrnnv2', 'SimVP', 'simvp', 'TAU', 'tau', 'MMVP', 'mmvp', 
                                'SwinLSTM', 'swinlstm', 'swinlstm_d', 'swinlstm_b'],
                        help='Name of video prediction method to train (default: "SimVP")')
    parser.add_argument('--config_file', '-c', default=None, type=str,
                        help='Path to the default config file')
    parser.add_argument('--model_type', default=None, type=str,
                        help='Name of model for SimVP (default: None)')
    parser.add_argument('--hid_S', type=int, default=64, help='Encoder/decoder spatial hidden channels')
    parser.add_argument('--hid_T', type=int, default=256, help='Mid translator hidden channels')
    parser.add_argument('--N_S', type=int, default=4, help='Number of spatial encoder/decoder stages')
    parser.add_argument('--N_T', type=int, default=8, help='Number of mid translator stages')
    # parser.add_argument('--mlp_ratio', type=float, default=8.0, help='MLP expansion ratio in Meta blocks')
    parser.add_argument('--spatio_kernel_enc', type=int, default=3, help='Encoder ConvSC kernel size')
    parser.add_argument('--spatio_kernel_dec', type=int, default=3, help='Decoder ConvSC kernel size')
    parser.add_argument('--act_inplace', action='store_true', default=False, help='Use inplace activations')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate(default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate for SimVP (default: 0.)')
    parser.add_argument('--use_inr_decoder', action='store_true', default=True,
                        help='Use LIIF INR decoder instead of Conv decoder')
    parser.add_argument('--enc_out_dim', type=int, default=4,
                        help='Encoder last conv output channels (bottleneck)')
    # parser.add_argument('--overwrite', action='store_true', default=False,
    #                     help='Whether to allow overwriting the provided config file with args')
    # INR / positional encoding
    parser.add_argument('--inr_hidden', type=int, nargs='+', default=[256, 256, 256])
    parser.add_argument('--posenc_num_freqs', type=int, default=6,
                        help='Number of Fourier frequency bands for coord PE')
    parser.add_argument('--posenc_include_input', action='store_true', default=False,
                        help='Concatenate raw coords with Fourier features')
    parser.add_argument('--coord_encoding', type=str, default='none', choices=['none','fourier','hashgrid'],
                        help='Type of coordinate encoding for INR decoder')
    parser.add_argument('--encode_rel_coords', action='store_true', default=True,
                        help='Encode relative coords (coord - q_coord) instead of absolute')
    parser.add_argument('--hash_num_levels', type=int, default=12)
    parser.add_argument('--hash_features_per_level', type=int, default=2)
    parser.add_argument('--hash_log2_hashmap_size', type=int, default=15)
    parser.add_argument('--hash_base_resolution', type=int, default=16)
    parser.add_argument('--hash_per_level_scale', type=float, default=2.0)

    # Loss configuration
    parser.add_argument('--rec_loss', type=str, default='mse', choices=['mse', 'l1', 'smoothl1'],
                        help='Reconstruction loss type for VAE or recon tasks')
    parser.add_argument('--pred_loss', type=str, default='mse', choices=['mse', 'l1', 'smoothl1'],
                        help='Prediction loss type for forecasting tasks')
    # VAE / ODE
    parser.add_argument('--vae', action='store_true', default=False,
                        help='Enable VAE-style training (reconstruction + KL)')
    parser.add_argument('--vae_beta', type=float, default=0.0,
                        help='Weight for KL divergence term')
    parser.add_argument('--pred_beta', type=float, default=1.0,
                        help='Weight for prediction loss when combining VAE+ODE')
    parser.add_argument('--use_neuralode', action='store_true', default=True,
                        help='Enable latent Neural ODE for multi-step prediction')
    parser.add_argument('--ode_hidden', type=int, default=64, help='Neural ODE hidden channels')
    parser.add_argument('--ode_layers', type=int, default=2, help='Neural ODE number of layers')
    parser.add_argument('--ode_method', type=str, default='rk4', help='Neural ODE solver method')
    parser.add_argument('--ode_dt', type=float, default=1.0, help='Neural ODE step size (dt)')
    parser.add_argument('--inr_chunk', type=int, default=None,
                        help='Optional chunk size for INR querying to save memory')
    parser.add_argument('--train_mask', type=float, default=0.1,
                        help='Fraction of coordinates to train on per image (0<r<=1). Applies only during training for INR decoding.')

    # Training parameters (optimizer)
    parser.add_argument('--epoch', '-e', default=600, type=int, help='end epochs (default: 200)')
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--no_display_method_info', action='store_true', default=False,
                        help='Do not display method info')

    # Training parameters (scheduler)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')

    # lightning
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--metric_for_bestckpt', default='val_loss', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument('--wandb_project', type=str, default='WINDY',
                        help='wandb project name (enable W&B logging)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='wandb entity (team/user), optional')

    return parser