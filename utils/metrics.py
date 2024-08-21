import math
import torch
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from models.components.loss import PearsonCorrLoss


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class ScalarMetricAccumulator(object):

    def __init__(self):
        super().__init__()
        self.accum_dict = {}
        self.count_dict = {}

    @torch.no_grad()
    def add(self, name, value, batchsize=None, mode=None):
        assert mode is None or mode in ('mean', 'sum')

        if mode is None:
            delta = value.sum()
            count = value.size(0)
        elif mode == 'mean':
            delta = value * batchsize
            count = batchsize
        elif mode == 'sum':
            delta = value
            count = batchsize
        delta = delta.item() if isinstance(delta, torch.Tensor) else delta

        if name not in self.accum_dict:
            self.accum_dict[name] = 0
            self.count_dict[name] = 0
        self.accum_dict[name] += delta
        self.count_dict[name] += count

    def log(self, it, tag, logger=BlackHole(), writer=BlackHole()):
        summary = {k: self.accum_dict[k] / self.count_dict[k] for k in self.accum_dict}
        logstr = '[%s] Iter %05d' % (tag, it)
        for k, v in summary.items():
            logstr += ' | %s %.4f' % (k, v)
            writer.add_scalar('%s/%s' % (tag, k), v, it)
        logger.info(logstr)

    def get_average(self, name):
        return self.accum_dict[name] / self.count_dict[name]


def per_complex_corr(df, pred_attr='y_pred', true_attr='y_true', limit=10):
    corr_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')
        if len(df_cplx) <= 2:
            continue
        # if len(df_cplx) < limit:
        #     continue
        y_pred = np.array(df_cplx[pred_attr])
        y_true = np.array(df_cplx[true_attr])
        # print("HIIII!", cal_pearson(y_pred, y_true))
        # print("Pred_and True:", y_pred, y_true)
        corr_table.append({
            'complex': cplx,
            'pearson': abs(cal_pearson(y_pred, y_true)),
            'spearman': abs(cal_spearman(y_pred, y_true)),
            'rmse': cal_rmse(y_pred, y_true),
            'mae': cal_mae(y_pred, y_true)
        })
    # print("Corr_tabel:", corr_table)
    corr_table = pd.DataFrame(corr_table)
    corr_table.fillna(0)
    avg = corr_table[['pearson', 'spearman', 'rmse', 'mae']].mean()
    return avg['pearson'], avg['spearman'], avg['rmse'], avg['mae']


def per_complex_acc(df, pred_attr='y_pred', true_attr='y_true', limit=10):
    acc_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')
        if len(df_cplx) <= 2:
            continue
        y_pred = np.array(df_cplx[pred_attr])
        y_true = np.array(df_cplx[true_attr])
        acc_table.append({
            'complex': cplx,
            'accuracy': cal_accuracy(y_pred, y_true),
            # 'auc': cal_auc(y_pred, y_true),
            'precision': cal_precision(y_pred, y_true),
            'recall': cal_recall(y_pred, y_true)
        })
    acc_table = pd.DataFrame(acc_table)
    # avg = acc_table[['accuracy', 'auc', 'precision', 'recall']].mean()
    # return avg['accuracy'], avg['auc'], avg['precision'], avg['recall']
    avg = acc_table[['accuracy', 'precision', 'recall']].mean()
    return avg['accuracy'], avg['precision'], avg['recall']


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def get_loss(loss_type, pred, y, reduction='none'):
    if loss_type == 'regression':
        # criterion = PearsonCorrLoss() 
        # losses = F.huber_loss(pred, y, delta=2, reduction=reduction)
        losses = F.mse_loss(pred, y, reduction=reduction)
        # print("MSE Loss:", F.mse_loss(pred, y, reduction=reduction))
        # print("Pearson Loss:", criterion(pred, y))
    elif loss_type == 'binary':
        losses = F.binary_cross_entropy_with_logits(pred, y, reduction=reduction)
    else:
        raise NotImplementedError("Loss Not Implemented!")
    return losses


def cal_weighted_loss(pred_dict, y, mask, loss_types, loss_weights):
    loss_list = []
    y_pred = pred_dict['y_pred']
    # print("Y_pred:", y_pred.shape, y.shape, mask.shape)
    if len(y_pred.shape) > 1:
        assert y_pred.shape[1] == len(loss_types)
    else:
        y_pred = y_pred.unsqueeze(1)
    for i, l_type in enumerate(loss_types):
        y_pred_i = y_pred[:, i]
        y_i = y[:, i]
        mask_i = mask[:, i]
        l_i = get_loss(l_type, y_pred_i, y_i) * float(loss_weights[i])
        if 'y_pred_inv' in pred_dict and l_type == 'regression' and i == 0:
            # Only ddG task has the inversion property
            y_pred_inv = pred_dict['y_pred_inv']
            if len(y_pred_inv.shape) == 1:
                y_pred_inv = y_pred_inv.unsqueeze(1)
            y_pred_inv_i = y_pred_inv[:, i]
            l_i_inv = get_loss(l_type, y_pred_inv_i, -y_i) * float(loss_weights[i])
            l_i = (l_i + l_i_inv) / 2
        loss_list.append(l_i)
    losses = torch.stack(loss_list, dim=-1)
    loss = (losses * mask).sum() / (mask.sum().clip(min=1))
    return loss
        

def cal_pearson(pred, gt):
    # print("Pearson Cal:", np.unique(pred.shape), np.unique(gt.shape))
    if np.isnan(stats.pearsonr(pred, gt).statistic):
        print("Pearson Cal:", pred, gt)
    return stats.pearsonr(pred, gt).statistic

def cal_spearman(pred, gt):
    if np.isnan(stats.spearmanr(pred, gt).statistic):
        print("SPearman Cal:", pred, gt)
    return stats.spearmanr(pred, gt).statistic

def cal_rmse(pred, gt):
    return math.sqrt(mean_squared_error(pred, gt))

def cal_mae(pred, gt):
    return np.abs(pred-gt).sum() / len(pred)

def cal_accuracy(pred, gt, thres=0.5):
    logits = 1 / (1+np.exp(-pred))
    binary = np.where(logits>=thres, 1, 0)
    acc = (binary == gt).sum() / np.size(binary)
    return acc

def cal_auc(pred, gt):
    logits = 1 / (1+np.exp(-pred))
    if len(np.unique(gt)) == 1:
        return 0
    return roc_auc_score(gt.astype(np.int32), logits)
    
def cal_precision(pred, gt, thres=0.5):
    logits = 1 / (1+np.exp(-pred))
    binary = np.where(logits>=thres, 1, 0)
    true_positives = (binary == 1) & (gt == 1)
    positive_preds = binary == 1
    return np.sum(true_positives) / np.sum(positive_preds) if np.sum(positive_preds) > 0 else 0
    
def cal_recall(pred, gt, thres=0.5):
    logits = 1 / (1+np.exp(-pred))
    binary = np.where(logits>=thres, 1, 0)
    true_positives = (binary == 1) & (gt == 1)
    actual_positives = (gt == 1)
    return np.sum(true_positives) / np.sum(actual_positives) if np.sum(actual_positives) > 0 else 0
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', default=None, type=str)
    args = parser.parse_args()
    df = pd.read_csv(args.pred_dir)
    pred = df['pred_0']
    gt = df['gt_0']
    pearson = cal_pearson(pred, gt)[0]
    spearman = cal_spearman(pred, gt)[0]
    rmse = cal_rmse(pred, gt)
    mae = cal_mae(pred, gt)
    print("Pearson Score is {}".format(pearson))
    print("Spearman Score is {}".format(spearman))
    print("RMSE Score is {}".format(rmse))
    print("MAE Score is {}".format(mae))
