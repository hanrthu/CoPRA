import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from models import ModelRegister
from utils.metrics import ScalarMetricAccumulator, cal_accuracy, cal_precision, cal_recall, cal_pearson, cal_spearman, \
                            cal_rmse, cal_mae, cal_weighted_loss, cal_auc, get_loss
                            
def get_model(model_args:dict=None):
    register = ModelRegister()
    model_args_ori = {}
    model_args_ori.update(model_args)
    model_cls = register[model_args['model_type']]
    model = model_cls(**model_args_ori)
    return model

class PretuneModule(pl.LightningModule):
    def __init__(self, output_dir=None, model_args=None, data_args=None, run_args=None):
        super().__init__()
        self.save_hyperparameters()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.l_type = data_args.loss_type
        self.model = get_model(model_args=model_args.model)
        self.model_args = model_args
        self.data_args = data_args
        self.run_args = run_args
        self.optimizers_cfg = self.model_args.train.optimizer
        self.scheduler_cfg = self.model_args.train.scheduler
        self.valid_it = 0
        self.temperature = model_args.train.temperature
        self.batch_size = data_args.batch_size

        self.train_loss = None

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        if self.optimizers_cfg.type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.optimizers_cfg.lr, 
                                         betas=(self.optimizers_cfg.beta1, self.optimizers_cfg.beta2, ))
        elif self.optimizers_cfg.type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optimizers_cfg.lr)
        elif self.optimizers_cfg.type == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.optimizers_cfg.lr)
        else:
            raise NotImplementedError('Optimizer not supported: %s' % self.optimizers_cfg.type)

        if self.scheduler_cfg.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor=self.scheduler_cfg.factor, 
                                                                   patience=self.scheduler_cfg.patience, 
                                                                   min_lr=self.scheduler_cfg.min_lr)
        elif self.scheduler_cfg.type == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                             milestones=self.scheduler_cfg.milestones, 
                                                             gamma=self.scheduler_cfg.gamma)
        elif self.scheduler_cfg.type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                               gamma=self.scheduler_cfg.gamma)
        else:
            raise NotImplementedError('Scheduler not supported: %s' % self.scheduler_cfg.type)

        if self.model_args.resume is not None:
            print("Resuming from checkloint: %s" % self.model_args.resume)
            ckpt = torch.load(self.model_args.resume, map_location=self.model_args.device)
            it_first = ckpt['iteration']
            lsd_result = self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
            print(
                'Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))

            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
            
        if self.scheduler_cfg.type == 'plateau':
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'val_loss'
                }
            }
        else:
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                }
            }
        return optim_dict

    def on_train_start(self):
        log_hyperparams = {'model_args':self.model_args, 'data_args': self.data_args, 'run_args': self.run_args}
        self.logger.log_hyperparams(log_hyperparams)

    def on_before_optimizer_step(self, optimizer) -> None:
        pass
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)
        #         print("Found Unused Parameters")
    
    def continuous_to_discrete_tensor(self, values):
        discrete_values = torch.zeros_like(values, dtype=torch.long, device=values.device)
        
        mask_0_8 = values < 8
        discrete_values[mask_0_8] = (values[mask_0_8] * 2 + 0.5).long()
        
        mask_8_32 = (values >= 8) & (values < 32)
        discrete_values[mask_8_32] = (16 + (values[mask_8_32] - 8)).long()
        
        mask_ge_32 = (values >= 32)
        discrete_values[mask_ge_32] = 39
        discrete_values = torch.clamp(discrete_values, 0, 39).long()
        return discrete_values
    
    def cal_loss(self, pred_clip, y_clip, pred_dist, y_dist, identifier):
        # y_dist = torch.clamp(y_dist.long(), 0, 31).long()
        y_dist = self.continuous_to_discrete_tensor(y_dist)
        total_y_dists = []
        total_pred_dists = []
        total_y_dists_inv = []
        total_pred_dists_inv = []
        for i in range(identifier.shape[0]):
            item_identifier = identifier[i].squeeze()
            y_tmp = y_dist[i, item_identifier==0]
            y_interface_dist = y_tmp[:, item_identifier==1].flatten()
            total_y_dists.append(y_interface_dist)
            
            y_pred_tmp = pred_dist[i, item_identifier==0]
            y_interface_pred = y_pred_tmp[:, item_identifier==1].reshape([-1, pred_dist.shape[-1]])
            total_pred_dists.append(y_interface_pred)
            
            y_tmp_inv = y_dist[i, item_identifier==1]
            y_interface_dist_inv = y_tmp_inv[:, item_identifier==0].flatten()
            total_y_dists_inv.append(y_interface_dist_inv)
            
            y_pred_tmp_inv = pred_dist[i, item_identifier==1]
            y_interface_pred_inv = y_pred_tmp_inv[:, item_identifier==0].reshape([-1, pred_dist.shape[-1]])
            total_pred_dists_inv.append(y_interface_pred_inv)
            
        total_pred_dists = torch.cat(total_pred_dists, dim=0)
        total_pred_dists_inv = torch.cat(total_pred_dists_inv, dim=0)
        total_y_dists = torch.cat(total_y_dists)
        total_y_dists_inv = torch.cat(total_y_dists_inv)
        # y_dist_one_hot = F.one_hot(indices, num_classes=32).float()
        loss_clip_forward = F.cross_entropy(pred_clip / self.temperature, y_clip.long())
        loss_clip_inverse = F.cross_entropy(pred_clip.transpose(0, 1) / self.temperature, y_clip.long())
        loss_clip = 0.5 * (loss_clip_forward + loss_clip_inverse)
        # print("Loss CLIP:", loss_clip)
        # loss_dist_forward = F.cross_entropy(y_interface_pred.permute(0, 3, 1, 2) / self.temperature , y_interface_dist.long())
        # loss_dist_inv = F.cross_entropy(y_interface_pred_inv.permute(0, 3, 1, 2) / self.temperature , y_interface_dist_inv.long())
        loss_dist_forward = F.cross_entropy(total_pred_dists / self.temperature , total_y_dists.long())
        loss_dist_inverse = F.cross_entropy(total_pred_dists_inv / self.temperature, total_y_dists_inv.long())
        
        loss_dist = 0.5 * (loss_dist_forward + loss_dist_inverse)
        # print("Loss Dist:", loss_dist)
        return loss_clip, loss_dist
    
    def training_step(self, batch, batch_idx):
        y_dist = batch['atom_min_dist']
        res_identifier = batch['identifier']
        # y_clip = batch['clip_label']
        y_clip = torch.arange(batch['size'], dtype=torch.long, device=y_dist.device)
        pred_dist, pred_clip = self.model(batch, self.data_args.strategy, stage='pretune', need_mask=True)
        loss_clip, loss_dist = self.cal_loss(pred_clip, y_clip, pred_dist, y_dist, res_identifier)
        loss = loss_clip + loss_dist
        if torch.isnan(loss).any():
            print("Found nan in loss!", input)
            exit()
        self.train_loss = loss.detach()
        self.log("train_loss", float(self.train_loss), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_clip_loss", float(loss_clip.detach()), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train_dist_loss", float(loss_dist.detach()), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.scalar_accum = ScalarMetricAccumulator()
        self.results = []

    def validation_step(self, batch, batch_idx):
        y_dist = batch['atom_min_dist']
        res_identifier = batch['identifier']
        y_clip = torch.arange(batch['size'], dtype=torch.long, device=y_dist.device)
        pred_dist, pred_clip = self.model(batch, self.data_args.strategy, stage='pretune', need_mask=True)
        
        loss_clip, loss_dist = self.cal_loss(pred_clip, y_clip, pred_dist, y_dist, res_identifier)
        val_loss = loss_clip + loss_dist
        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=batch['size'], mode='mean')
        self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_clip_loss", float(loss_clip.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val_dist_loss", float(loss_dist.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return val_loss
    
    def on_validation_epoch_end(self):
        val_loss = self.scalar_accum.get_average('val_loss')
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        # Trigger scheduler
        self.valid_it += 1
        return val_loss

    def on_test_epoch_start(self) -> None:
        self.results = []
        self.scalar_accum = ScalarMetricAccumulator()
        
    def test_step(self, batch, batch_idx):
        y_dist = batch['atom_min_dist']
        res_identifier = batch['identifier']
        y_clip = torch.arange(batch['size'], dtype=torch.long, device=y_dist.device)
        pred_dist, pred_clip = self.model(batch, self.data_args.strategy, stage='pretune', need_mask=True)
        loss_clip, loss_dist = self.cal_loss(pred_clip, y_clip, pred_dist, y_dist, res_identifier)
        test_loss = loss_clip + loss_dist
        self.scalar_accum.add(name='loss', value = test_loss, batchsize=batch['size'], mode='mean')
        self.log("test_loss_step", test_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_clip_loss", float(loss_clip.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_dist_loss", float(loss_dist.detach()), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.res = {"test_loss_step": float(test_loss.detach()),"test_clip_loss": float(loss_clip.detach()), "test_dist_loss": float(loss_dist.detach())}
        return test_loss

    def on_test_epoch_end(self):
        test_loss = self.scalar_accum.get_average('loss')
        self.log('test_loss', test_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        
        return test_loss