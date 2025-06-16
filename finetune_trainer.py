import torch
from tqdm import tqdm
import torch
from finetune_evaluator import Evaluator
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from timeit import default_timer as timer
import numpy as np
import copy
import os
from argparse import Namespace


class Trainer(object):
    """
    Pytorch Trainer for fine-tuning CBraMod on downstream tasks.

    Attributes
    ----------
    params : argparse.Namespace
        Parameters for the training process.
    data_loader : dict
        Dictionary containing data loaders for training, validation, and testing.
    model : torch.nn.Module
        The model (backbone + downstream task head) to be trained.
    criterion : torch.nn.Module
        Loss function used for training, varies by task type.
    best_model_states : dict
        Best model states based on validation performance.
    optimizer : torch.optim.Optimizer
        Optimizer for training the model.
    optimizer_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler for the optimizer.
    data_length : int
        Length of the training data loader, used for scheduling.
    """

    def __init__(
            self,
            params: Namespace,
            data_loader: torch.utils.data.DataLoader,
            model: torch.nn.Module):
        """
        Parameters
        ----------
        params : argparse.Namespace
            Parameters for the training process.
        data_loader : dict
            Dictionary containing data loaders for
            training, validation, and testing.
        model : torch.nn.Module
            The model (backbone + downstream task head) to be trained.
        """
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.task_type == 'multiclass':
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.task_type == 'binaryclass':
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.task_type == 'regression':
            self.criterion = MSELoss().cuda()
        else:
            raise ValueError(
                'task_type must be one of [multiclass, binaryclass, regression], '
                ' but got {}'.format(self.params.task_type))

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print('Model Architecture: ')
        print(self.model)
        print('Number of learnable parameters: ', 
              sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = -1
        acc_best = 0

        metric_names = ['Accuracy', "Cohen's kappa", "F1 score", "Learning rate"]

        global_start_time = timer()

        if self.params.progress_bar:
            loader = tqdm(self.data_loader['train'], mininterval=10)
        else:
            loader = self.data_loader['train']

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in loader:
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, {}: {:.5f}, {}: {:.5f}, "
                     "{}: {:.5f}, {}: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        metric_names[0],
                        acc,
                        metric_names[1],
                        kappa,
                        metric_names[2],
                        f1,
                        metric_names[3],
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print('Confusion Matrix:')
                print(cm)

                # Validation criteria: lower Cohen's Kappa
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        print('Total Fine-tuning time ', (timer() - global_start_time) / 60, 'mins')
        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states)

        # Evaluation
        with torch.no_grad():
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(
                    metric_names[0],
                    acc,
                    metric_names[1],
                    kappa,
                    metric_names[2],
                    f1,
                )
            )
            print('Confusion Matrix: ')
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(
                best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0

        metric_names = ['Accuracy', 'PR AUC', 'ROC AUC', 'Learning rate']

        if self.params.progress_bar:
            loader = tqdm(self.data_loader['train'], mininterval=10)
        else:
            loader = self.data_loader['train']

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in loader:
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, {}: {:.5f}, {}: {:.5f}, "
                    "{}: {:.5f}, {}: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        metric_names[0],
                        acc,
                        metric_names[1],
                        pr_auc,
                        metric_names[2],
                        roc_auc,
                        metric_names[3],
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print('Confusion Matrix: ')
                print(cm)
                if roc_auc > roc_auc_best:
                    print("Area Under Curve of ROC increasing....saving weights !! ")
                    print("Val Evaluation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(
                        metric_names[0],
                        acc,
                        metric_names[1],
                        pr_auc,
                        metric_names[2],
                        roc_auc,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states)

        with torch.no_grad():
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(
                    metric_names[0],
                    acc,
                    metric_names[1],
                    pr_auc,
                    metric_names[2],
                    roc_auc,
                )
            )

            print('Confusion Matrix:')
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(
                best_f1_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states)

        with torch.no_grad():
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
