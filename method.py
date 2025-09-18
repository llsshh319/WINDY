import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os.path as osp
import lightning as l
from utils.main_utils import print_log, check_dir
from core import get_optim_scheduler, timm_schedulers
from core import metric
from model import WINDY_Model


class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters(args)
        self.model = self._build_model(**args)
        def make_loss(name: str):
            name = (name or 'mse').lower()
            if name == 'mse':
                return nn.MSELoss()
            if name == 'l1':
                return nn.L1Loss()
            if name == 'smoothl1':
                return nn.SmoothL1Loss(beta=0.1)
            return nn.MSELoss()

        # default prediction criterion (used in base validation/test)
        self.criterion = make_loss(args.get('pred_loss', 'mse'))
        # optional reconstruction criterion for methods that use it
        self.rec_criterion = make_loss(args.get('rec_loss', 'mse'))
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        # Log validation loss aggregated per-epoch and show on the progress bar
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
            channel_names=self.channel_names, spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None))
        
        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        return results_all

class WINDY(Base_method):

    def __init__(self, **args):
        super().__init__(**args)
        self.vae_beta = float(args.get('vae_beta', 0.0))
        self.pred_beta = float(args.get('pred_beta', 1.0))
        self.vae = bool(args.get('vae', self.vae_beta > 0.0))
        self.use_neuralode = bool(args.get('use_neuralode', False))

    def _build_model(self, **args):
        # Ensure the model knows if VAE mode is enabled
        args = {**args, 'vae': bool(args.get('vae', args.get('vae_beta', 0.0) > 0.0))}
        return WINDY_Model(**args)
        
    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if self.vae and not self.use_neuralode:
            # In VAE-only mode, reconstruct the input sequence
            return self.model(batch_x)
        if self.use_neuralode:
            # Generate aft_seq_length predictions using latent ODE
            pred_y = self.model(batch_x)
            return pred_y
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    
    def _masked_loss(self, out_dict, target_seq, criterion):
        # out_dict: {'pred': (N,Q,C), 'mask': (HW,), optional 'B','T_out'}
        pred_q = out_dict['pred']          # (N, Q, C)
        mask_flat = out_dict['mask']       # (HW,) bool
        B = out_dict.get('B', None)
        T_out = out_dict.get('T_out', None)
        if B is not None and T_out is not None:
            target = target_seq[:, :T_out]  # (B,T,C,H,W)
            B, T, C, H, W = target.shape
            target = target.reshape(B * T, C, H, W)
        else:
            # assume target already flattened batch-time
            bt, C, H, W = target_seq.shape
            target = target_seq
        # flatten spatial and select by mask
        target_flat = target.permute(0, 2, 3, 1).reshape(target.shape[0], H * W, C)  # (N, HW, C)
        target_q = target_flat[:, mask_flat, :]  # (N, Q, C)
        # compute elementwise loss
        from torch import nn
        if isinstance(criterion, nn.MSELoss):
            loss_map = (pred_q - target_q) ** 2
        elif isinstance(criterion, nn.L1Loss):
            loss_map = (pred_q - target_q).abs()
        elif isinstance(criterion, nn.SmoothL1Loss):
            beta = getattr(criterion, 'beta', 1.0)
            diff = (pred_q - target_q).abs()
            loss_map = torch.where(diff < beta, 0.5 * diff.pow(2) / beta, diff - 0.5 * beta)
        else:
            loss_map = (pred_q - target_q) ** 2
        return loss_map.mean()

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        if self.vae and not self.use_neuralode:
            recon_y, kl = self.model(batch_x)
            if isinstance(recon_y, dict):
                rec_loss = self._masked_loss(recon_y, batch_x, self.rec_criterion)
            else:
                rec_loss = self.rec_criterion(recon_y, batch_x)
            loss = rec_loss + self.vae_beta * kl
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_rec', rec_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log('train_kl', kl, on_step=True, on_epoch=True, prog_bar=False)
            return loss
        elif self.vae and self.use_neuralode:
            # Joint VAE + ODE: reconstruction on input + prediction on target + KL
            # recon_y, kl = self.model(batch_x)
            if isinstance(recon_y, dict):
                rec_loss = self._masked_loss(recon_y, batch_x, self.rec_criterion)
            else:
                rec_loss = self.rec_criterion(recon_y, batch_x)
            pred_y, kl = self.model(batch_x)
            if isinstance(pred_y, dict):
                pred_loss = self._masked_loss(pred_y, batch_y, self.criterion)
            else:
                pred_loss = self.criterion(pred_y, batch_y)
            loss = rec_loss + self.vae_beta * kl + self.pred_beta * pred_loss
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_rec', rec_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log('train_pred', pred_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log('train_kl', kl, on_step=True, on_epoch=True, prog_bar=False)
            return loss
        else:
            pred_y = self.model(batch_x)
            if isinstance(pred_y, dict):
                loss = self._masked_loss(pred_y, batch_y, self.criterion)
            else:
                loss = self.criterion(pred_y, batch_y)
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        if self.vae and not self.use_neuralode:
            recon_y, kl = self.model(batch_x)
            if isinstance(recon_y, dict):
                rec_loss = self._sparse_loss(recon_y, batch_x, self.rec_criterion)
            else:
                rec_loss = self.rec_criterion(recon_y, batch_x)
            loss = rec_loss + self.vae_beta * kl
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        else:
            pred_y = self.model(batch_x)
            if isinstance(pred_y, dict):
                loss = self._sparse_loss(pred_y, batch_y, self.criterion)
            else:
                loss = self.criterion(pred_y, batch_y)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred = self.model(batch_x)
        if isinstance(pred, dict):
            # compute loss only
            is_recon = self.vae and not self.use_neuralode
            target = batch_x if is_recon else batch_y
            crit = self.rec_criterion if is_recon else self.criterion
            loss = self._masked_loss(pred, target, crit)
            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return {'loss': loss}
        else:
            outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred.detach().cpu().numpy(), 'trues': batch_y.cpu().numpy()}
            self.test_outputs.append(outputs)
            return outputs