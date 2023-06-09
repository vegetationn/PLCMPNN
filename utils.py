import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from rdkit import Chem
from sklearn import metrics
from sklearn.model_selection import train_test_split
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.optim.lr_scheduler import _LRScheduler


def seed_everything(seed=666):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def initialize_weights(model):
    """
    Initializes the weights of a model in place.
    :param model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


def scaffold_split(smile_list, frac=None, balanced=False, include_chirality=False, ramdom_state=0):
    frac = [0.8, 0.1, 0.1] if frac is None else frac
    assert sum(frac) == 1

    n_total_valid = int(np.floor(frac[1] * len(smile_list)))
    n_total_test = int(np.floor(frac[2] * len(smile_list)))
    n_total_train = len(smile_list) - n_total_valid - n_total_test

    scaffolds_sets = defaultdict(list)
    for idx, smile in enumerate(smile_list):
        mol = Chem.MolFromSmiles(smile)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        scaffolds_sets[scaffold].append(idx)

    # Put stuff that's bigger than half the val/test size into train, rest just order randomly
    if balanced:
        index_sets = list(scaffolds_sets.values())
        big_index_sets, small_index_sets = list(), list()
        for index_set in index_sets:
            if len(index_set) > n_total_valid / 2 or len(index_set) > n_total_test / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(ramdom_state)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffolds_sets.values()), key=lambda index_set: len(index_set), reverse=True)

    train_index, valid_index, test_index = list(), list(), list()
    for index_set in index_sets:
        if len(train_index) + len(index_set) <= n_total_train:
            train_index += index_set
        elif len(valid_index) + len(index_set) <= n_total_valid:
            valid_index += index_set
        else:
            test_index += index_set

    return [smile_list[idx] for idx in train_index], [smile_list[idx] for idx in valid_index], [smile_list[idx] for idx in test_index]


def random_split(smile_list, frac=None, random_state=0):
    frac = [0.8, 0.1, 0.1] if frac is None else frac
    assert sum(frac) == 1

    n_total_valid = int(np.floor(frac[1] * len(smile_list)))
    n_total_test = int(np.floor(frac[2] * len(smile_list)))
    n_total_train = len(smile_list) - n_total_valid - n_total_test

    train_valid_smiles, test_smiles = train_test_split(smile_list, test_size=n_total_test, random_state=random_state)
    train_smiles, valid_smiles = train_test_split(train_valid_smiles, test_size=n_total_valid, random_state=random_state)

    return train_smiles, valid_smiles, test_smiles


def cal_loss(y_true, y_pred, loss_name, data_mean, data_std, device):
    # y_true, y_pred.shape = tensor shape with (batch, task_number), data_mean, data_std.shape = tensor shape with (1, task_number)
    y_true = torch.true_divide((y_true - data_mean), data_std)
    if loss_name == 'mse':
        # convert labels to float
        y_true = y_true.float()
        loss = F.mse_loss(y_pred, y_true, reduction="sum") / y_true.shape[1]
    elif loss_name == 'bce':
        # convert labels to long
        y_true = y_true.long()
        # find all -1 in y_true
        y_mask = torch.where(y_true == -1, torch.tensor([0]).to(device), torch.tensor([1]).to(device))
        y_cal_true = torch.where(y_true == -1, torch.tensor([0]).to(device), y_true)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_cal_true.float(), reduction='none') * y_mask
        loss = loss.sum() / y_true.shape[1]
    else:
        raise "please refer loss function!"
    return loss


def cal_metric(y_true, y_pred, metric_name, data_mean, data_std):
    # y_true, y_pred.shape = numpy shape with (batch, task_number), data_mean, data_std.shape = numpy shape with (1, task_number)
    y_pred = y_pred * data_std + data_mean
    if metric_name == 'rmse':
        metric = np.sqrt(np.nanmean(np.square(y_pred - y_true)))
    elif metric_name == 'mae':
        metric = np.nanmean(np.abs(y_pred - y_true))
    elif metric_name == 'auc':
        # convert labels to long
        y_true = y_true.astype(np.int64)
        score_list = list()
        for task_idx in range(y_true.shape[1]):
            true, pred = y_true[:, task_idx], y_pred[:, task_idx]
            # only calculate the label 0's and 1's, ignore label -1's.
            true, pred = true[np.where(true >= 0)], pred[np.where(true >= 0)]
            # if all 0's or all 1's, append nan in multitask labels, raise error in single task label.
            if len(set(true)) == 1:
                if y_true.shape[1] > 1:
                    score_list.append(float('nan'))
                else:
                    raise "the single task label are all 0's or 1's!"
            else:
                score_list.append(metrics.roc_auc_score(y_true=true, y_score=pred))
        metric = np.nanmean(score_list)
    elif metric_name == 'prc-auc':
        # convert labels to long
        y_true = y_true.astype(np.int64)
        score_list = list()
        for task_idx in range(y_true.shape[1]):
            true, pred = y_true[:, task_idx], y_pred[:, task_idx]
            # only calculate the label 0's and 1's, ignore label -1's.
            true, pred = true[np.where(true >= 0)], pred[np.where(true >= 0)]
            # if all 0's or all 1's, append nan in multitask labels, raise error in single task label.
            if len(set(true)) == 1:
                if y_true.shape[1] > 1:
                    score_list.append(float('nan'))
                else:
                    raise "the single task label are all 0's or 1's!"
            else:
                precision, recall, _ = metrics.precision_recall_curve(y_true=true, probas_pred=pred)
                score_list.append(metrics.auc(recall, precision))
        metric = np.nanmean(score_list)
    else:
        raise "please refer metric function!"
    return metric


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, init_lr, max_lr, final_lr):
        """
        Initializes the learning rate scheduler.
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]
