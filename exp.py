# Copyright (c) CAIRI AI Lab. All rights reserved

import sys
import time
import os.path as osp
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch

import lightning as l

from method import WINDY
# from openstl.datasets.base_data import BaseDataModule
from dataloader import get_dataset
from callbacks import SetupCallback, EpochEndCallback, BestCheckpointCallback
from utils import measure_throughput
from lightning import seed_everything, Trainer
import lightning.pytorch.callbacks as lc
from lightning.pytorch.loggers import WandbLogger


class BaseDataModule(l.LightningDataModule):
    def __init__(self, train_loader, valid_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.test_mean = test_loader.dataset.mean
        self.test_std = test_loader.dataset.std
        self.data_name = test_loader.dataset.data_name

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args, dataloaders=None, strategy='auto'):
        """Initialize experiments (non-dist as an example)"""
        self.args = args

        base_dir = args.res_dir if args.res_dir is not None else 'work_dirs'
        save_dir = osp.join(base_dir, args.ex_name if not args.ex_name.startswith(args.res_dir) \
            else args.ex_name.split(args.res_dir+'/')[-1])
        ckpt_dir = osp.join(save_dir, 'checkpoints')

        seed_everything(args.seed)
        self.data = self._get_data(dataloaders)
        # Derive in_shape and sensible defaults from a real
        bx, _ = next(iter(self.data.train_loader))
        # bx: (B, T, C, H, W)
        _, T, C, H, W = bx.shape
        self.args.in_shape = [int(T), int(C), int(H), int(W)]
        
        self.method = WINDY(steps_per_epoch=len(self.data.train_loader), \
            test_mean=self.data.test_mean, test_std=self.data.test_std, save_dir=save_dir, **self.args.__dict__)
        callbacks, self.save_dir = self._load_callbacks(save_dir, ckpt_dir)
        # Choose strategy: use DDP explicitly if requested and multiple devices
        requested_strategy = 'ddp' if (self.args.dist and isinstance(self.args.gpus, (list, tuple)) and len(self.args.gpus) > 1) else strategy
        self.trainer = self._init_trainer(self.args, callbacks, requested_strategy)

    def _init_trainer(self, args, callbacks, strategy):
        # Lightning accepts either list of indices or int count; we pass list
        logger = WandbLogger(
            name=args.ex_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_dir=self.save_dir,
            log_model=False,
        )
        return Trainer(
            devices=args.gpus,
            max_epochs=args.epoch,
            strategy=strategy,
            accelerator='gpu',
            callbacks=callbacks,
            num_nodes=1,
            # Avoid hanging in DDP when using subset of visible GPUs
            default_root_dir=self.save_dir,
            logger=logger,
        )

    def _load_callbacks(self, save_dir, ckpt_dir):
        method_info = None
        if self.args.dist == 0:
            if not self.args.no_display_method_info:
                method_info = self.display_method_info(self.args)

        setup_callback = SetupCallback(
            prefix = 'train' if (not self.args.test) else 'test',
            setup_time = time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            save_dir = save_dir,
            ckpt_dir = ckpt_dir,
            args = self.args,
            method_info = method_info,
            argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],
        )

        ckpt_callback = BestCheckpointCallback(
            monitor=self.args.metric_for_bestckpt,
            filename=f"best-{{epoch:02d}}-{{{self.args.metric_for_bestckpt}:.3f}}",
            mode='min',
            save_last=True,
            dirpath=ckpt_dir,
            verbose=True,
            every_n_epochs=self.args.log_step,
        )
        
        epochend_callback = EpochEndCallback()

        callbacks = [setup_callback, ckpt_callback, epochend_callback]
        if self.args.sched:
            callbacks.append(lc.LearningRateMonitor(logging_interval=None))
        return callbacks, save_dir

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            train_loader, vali_loader, test_loader = \
                get_dataset(**self.args.__dict__)
        else:
            train_loader, vali_loader, test_loader = dataloaders

        vali_loader = test_loader if vali_loader is None else vali_loader
        return BaseDataModule(train_loader, vali_loader, test_loader)

    def train(self):
        self.trainer.fit(self.method, self.data, ckpt_path=self.args.ckpt_path if self.args.ckpt_path else None)

    def test(self):
        if self.args.test == True:
            ckpt = torch.load(osp.join(self.save_dir, 'checkpoints', 'best.ckpt'))
            self.method.load_state_dict(ckpt['state_dict'])
        self.trainer.test(self.method, self.data)
    
    def display_method_info(self, args):
        """Plot the basic infomation of supported methods"""
        device = torch.device(args.device)
        if args.device == 'cuda':
            assign_gpu = 'cuda:' + (str(args.gpus[0]) if len(args.gpus) == 1 else '0')
            device = torch.device(assign_gpu)
        T, C, H, W = args.in_shape
        input_dummy = torch.ones(1, args.pre_seq_length, C, H, W).to(device)

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = ""
        try:
            flops = FlopCountAnalysis(self.method.model.to(device), input_dummy)
            flops = flop_count_table(flops)
        except Exception:
            flops = "(FLOPs estimation skipped)"

        if args.fps:
            fps = measure_throughput(self.method.model.to(device), input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(args.method, fps)
        else:
            fps = ''
        return info, flops, fps, dash_line
