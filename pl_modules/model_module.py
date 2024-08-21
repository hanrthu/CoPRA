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

class ModelModule(pl.LightningModule):
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

    def training_step(self, batch, batch_idx):
        y = batch['labels']
        pred = self.model(batch, self.data_args.strategy)
        # print(y.shape, pred.shape)
        loss = get_loss(self.l_type, pred, y, reduction='mean')
        # if torch.isnan(loss).any():
        #     print("Found nan in loss!", input)
        #     exit()
        self.train_loss = loss.detach()
        self.log("train_loss", float(self.train_loss), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.scalar_accum = ScalarMetricAccumulator()
        self.results = []

    def validation_step(self, batch, batch_idx):
        y = batch['labels']
        pred = self.model(batch, self.data_args.strategy)
        val_loss = get_loss(self.l_type, pred, y, reduction='mean')
        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=self.batch_size, mode='mean')
        self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for  y_true, y_pred in zip(batch['labels'], pred):
            result = {}
            result['y_true'] = y_true.item()
            result['y_pred'] = y_pred.item()
            self.results.append(result)
        return val_loss
    
    def on_validation_epoch_end(self):
        results = pd.DataFrame(self.results)
        # print("Validation:", results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_{self.valid_it}.csv'), index=False)
        if self.l_type == 'regression':
            y_pred = np.array(results[f'y_pred'])
            y_true = np.array(results[f'y_true'])
            pearson_all = np.abs(cal_pearson(y_pred, y_true))
            spearman_all = np.abs(cal_spearman(y_pred, y_true))
            rmse_all = cal_rmse(y_pred, y_true)
            mae_all = cal_mae(y_pred, y_true)
            print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
            
            self.log(f'val/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)

        elif self.l_type == 'binary':
            y_pred = np.array(results['y_pred'])
            y_true = np.array(results['y_true'])
            acc_all = cal_accuracy(y_pred, y_true)
            auc_all = cal_auc(y_pred, y_true)
            precision_all = cal_precision(y_pred, y_true)
            recall_all = cal_recall(y_pred, y_true)
            print(f'[All_Task] ACC {acc_all:.6f} PRECISION {precision_all:.6f} RECALL {recall_all:.6f}')

            self.log(f'val/all_acc', acc_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/all_auc', auc_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/all_precision', precision_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'val/all_recall', recall_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
    
        val_loss = rmse_all * rmse_all
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        # Trigger scheduler
        self.valid_it += 1
        return val_loss

    def on_test_epoch_start(self) -> None:
        self.results = []
        self.scalar_accum = ScalarMetricAccumulator()
        
    def test_step(self, batch, batch_idx):
        y = batch['labels']
        pred = self.model(batch, self.data_args.strategy)
        test_loss = get_loss(self.l_type, pred, y, reduction='mean')
        self.scalar_accum.add(name='loss', value = test_loss, batchsize=self.batch_size, mode='mean')
        self.log("test_loss_step", test_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for y_true, y_pred in zip(batch['labels'], pred):
            result = {}
            result['y_true'] = y_true.item()
            result['y_pred'] = y_pred.item()
            self.results.append(result)
        return test_loss

    def on_test_epoch_end(self):
        results = pd.DataFrame(self.results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_test.csv'), index=False)
        if self.l_type == 'regression':
            y_pred = np.array(results[f'y_pred'])
            y_true = np.array(results[f'y_true'])
            pearson_all = np.abs(cal_pearson(y_pred, y_true))
            spearman_all = np.abs(cal_spearman(y_pred, y_true))
            rmse_all = cal_rmse(y_pred, y_true)
            mae_all = cal_mae(y_pred, y_true)
            print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
            
            self.log(f'test/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'test/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'test/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'test/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.res = {"pearson": pearson_all,"spearman": spearman_all, "rmse": rmse_all, "mae": mae_all}
        elif self.l_type == 'binary':
            y_pred = np.array(results[f'y_pred'])
            y_true = np.array(results[f'y_true'])
            acc_all = cal_accuracy(y_pred, y_true)
            auc_all = cal_auc(y_pred, y_true)
            precision_all = cal_precision(y_pred, y_true)
            recall_all = cal_recall(y_pred, y_true)
            print(f'[All_Task] ACC {acc_all:.6f} PRECISION {precision_all:.6f} RECALL {recall_all:.6f} AUC {auc_all:.6f}')

            self.log(f'test/all_acc', acc_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'test/all_auc', auc_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'test/all_precision', precision_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.log(f'test/all_recall', recall_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
            self.res = {"acc": acc_all,"precision": precision_all, "recall": rmse_all, "auc": auc_all}
        print("Self.Res:", self.res)
        # test_loss = self.scalar_accum.get_average('loss')
        test_loss = rmse_all * rmse_all
        self.log('test_loss', test_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        
        return test_loss