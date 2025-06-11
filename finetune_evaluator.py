import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, confusion_matrix,
    cohen_kappa_score, roc_auc_score,
    precision_recall_curve, auc, r2_score, mean_squared_error
)
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import Namespace


class Evaluator:
    """
    Evaluator class to compute various metrics for model evaluation. 
    """
    def __init__(self, params: Namespace, data_loader: DataLoader):
        """
        Parameters
        ----------
        params : argparse.Namespace
            Placeholder, can be used for future configurations or parameters.
        data_loader : torch.utils.data.DataLoader
            Provide batches of data for evaluation.
        """
        # self.params = params 
        self.data_loader = data_loader

    def get_metrics_for_multiclass(
            self, model: torch.nn.Module
        ) -> tuple:
        """
        Evaluate the model on a multiclass classification task.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.

        Returns
        -------
        tuple
            A tuple containing accuracy, kappa score, F1 score, and confusion matrix.
            Each array is computed from the model's predictions on the data loader.
            Accuracy, kappa score, and F1 score are scalars,
            while confusion matrix is a 2D array of shape (num_classes, num_classes).
        """
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)
        return acc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model: torch.nn.Module):
        """
        Evaluate the model on a binary classification task.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        
        Returns
        -------
        acc : float
            Accuracy of the model predictions.
        pr_auc : float
            Area under the precision-recall curve.
        roc_auc : float
            Area under the receiver operating characteristic (ROC) curve.
        cm : np.ndarray
            Confusion matrix of the model predictions, shape (2, 2).
        """
        model.eval()

        truths = []
        preds = []
        scores = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, _ = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(
            self, model: torch.nn.Module
        ) -> tuple:
        """
        Evaluate the model on a regression task.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.

        Returns
        -------
        corrcoef : float
            Pearson correlation coefficient between true and predicted values.
        r2 : float
            Coefficient of determination (R^2) score.
        rmse : float
            Root mean squared error (RMSE) of the predictions.
        """
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse