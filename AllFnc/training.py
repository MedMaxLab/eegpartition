from __future__ import annotations
from collections import OrderedDict
from collections.abc import Iterable, Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
from scipy.stats import zscore
import seaborn as sns
import selfeeg.losses as Loss
from selfeeg.ssl import evaluate_loss
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
)
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
from typing import Optional, Union
from .utilities import GetLrDict

__all__ = [
    'train_model',
    'loadEEG',
    'lossBinary',
    'lossMulti',
    'get_performances',
    'GetLearningRate'
]

def loadEEG(path: str, 
            return_label: bool=True, 
            downsample: bool=False,
            use_only_original: bool= False,
            apply_zscore: bool = True,
            onehot_label: bool = False
           ):
    '''
    ``loadEEG`` loads the entire EEG signal stored in path.
    It is supposed to load pickle files with names 
    
        {dataset_ID}_{subject_ID}_{session_ID}_{object_ID}.pickle
    
    where each file contains a dictionary with keys:
        
        - 'data'  : for the signal.
        - 'label' : for the label. 

    Parameters
    ----------
    path: str
        The full path to the pickle file.
    return_label: bool, optional
        Whether to return the label or not. The function GetEEGPartitionNumber
        doesn't want a label. That's why we added the option to omit it.
        Default = True
    downsample: bool, optional
        Whether to downsample the EEG data to 125 Hz or not. Note that all files
        are supposed to have 250 Hz, since they come from the BIDSAlign preprocessing
        pipeline presented in the paper.
        Default = False
    use_only_original: bool, optional
        Whether to use only the original EEG channels or not. BIDSAlign apply a 
        template alignment, which included a spherical interpolation of channels not
        included in the library's 10_10 61 channels template.
        Default = False
    apply_zscore: bool, optional
        Whether to apply the z-score on each channel or not. 
        Default = True

    Returns
    -------
    x: Arraylike
        The arraylike object with the entire eeg signal to be partitioned by the 
        Pytorch's Dataset class (or whatever function is assigned for such task)
    y: float
        A float value with the EEG label.
    
    '''
    
    # NOTE: files were converted in pickle with the 
    # MatlabToPickle Jupyter Notebook. 
    with open(path, 'rb') as eegfile:
        EEG = pickle.load(eegfile)

    # extract and adapt data to training setting
    x = EEG['data']

    # get the dataset ID to coordinate some operations
    data_id = int(path.split(os.sep)[-1].split('_')[0])
    
    # if 125 Hz take one sample every 2
    if downsample:
        if data_id == 25:
            pass
        elif data_id == 99:
            pass
        else:
            x = x[:,::2]
    
    # if use original, interpolated channels are removed.
    # Check the dataset_info.json in each Summary folder file 
    # to know which channel was interpolated during the preprocessing
    if use_only_original:
        if data_id == 2:
            chan2dele = [34,44]
        elif data_id == 10:
            chan2dele = [ 1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 15, 
                         16, 17, 18, 19, 21, 23, 24, 26, 27, 29, 30, 
                         32, 33, 34, 36, 38, 40, 41, 42, 43, 44, 46, 
                         48, 50, 51, 52, 53, 54, 56, 58, 59]
        elif data_id == 20:
            chan2dele = [34, 44]
        elif data_id == 7:
            chan2del = [19, 27, 29, 39, 54]
        elif data_id == 19:
            chan2dele = [28, 30]
        elif data_id == 25:
            chan2dele = []
        elif data_id == 99:
            chan2dele = []
        else:
            chan2dele = [ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 
                         23, 27, 29, 30, 32, 34, 36, 38, 40, 42, 44, 
                         46, 48, 50, 52, 54, 56, 58]
        x = np.delete(x, chan2dele, 0)

    # apply z-score on the channels.
    if apply_zscore:
        x = zscore(x,1)

    # GetEEGPartitionNumber doesn't want a label, so we need to add a function
    # to omit the label
    if return_label:
        y = EEG['label']
        # one hot is needed if multiclass classification is performed
        if onehot_label and data_id == 10:
            y = F.one_hot(y, num_classes = 3)
        else:
            y = float(y)
        return x, y
    else:
        return x


def GetLearningRate(model, task):
    lr_dict = GetLrDict()
    lr = lr_dict.get(model).get(task)
    return lr


def lossBinary(yhat, ytrue):
    '''
    Just an alias to the binary_cross_entropy_with_logits function.
    Remember that yhat must be a tensor with the model output in the logit form,
    so no sigmoid operator should be applied on the model's output. 
    Remember that ytrue must be a float tensor with the same size as yhat and 
    with 0 or 1 based on the binary class.
    '''
    yhat = yhat.flatten()
    return F.binary_cross_entropy_with_logits(yhat, ytrue)


def lossMulti(yhat, ytrue):
    '''
    Just an alias to the binary_cross_entropy_with_logits function.
    Remember that yhat must be a tensor with the model output in the logit form,
    so no sigmoid operator should be applied on the model's output. 
    Remember that ytrue must be a float tensor with the same size as yhat and 
    with 0 or 1 based on the true class (e.g., [[0.,1.,0.], [1.,0.,0.], [0.,0.,1.]])
    Alternatively, it must be a long tensor with the class index (e.g., [1,0,2])
    '''
    return F.cross_entropy(yhat, ytrue)

def train_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    epochs=1,
    optimizer=None,
    augmenter=None,
    loss_func: Callable or list[Callable] = None,
    loss_args: list or dict = [],
    validation_loss_func: Callable or list[Callable] = None,
    validation_loss_args: list or dict = [],
    label_encoder: Callable or list[Callable] = None,
    lr_scheduler=None,
    EarlyStopper=None,
    validation_dataloader: torch.utils.data.DataLoader = None,
    verbose=True,
    device: str or torch.device = None,
    return_loss_info: bool = False,
) -> Optional[dict]:
    """
    copy of selfeeg.ssl.fine_tune function with the possibility to give
    a different validation loss function (and args)
    """

    if device is None:
        device = torch.device("cpu")
    else:
        if isinstance(device, str):
            device = torch.device(device.lower())
        elif isinstance(device, torch.device):
            pass
        else:
            raise ValueError("device must be a string or a torch.device instance")
    model.to(device=device)

    if not (isinstance(train_dataloader, torch.utils.data.DataLoader)):
        raise ValueError("Current implementation accept only training data as a pytorch DataLoader")
    if not (isinstance(epochs, int)):
        epochs = int(epochs)
    if epochs < 1:
        raise ValueError("epochs must be bigger than 1")
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    if loss_func is None:
        raise ValueError("loss function not given")
    if not (isinstance(loss_args, list) or isinstance(loss_args, dict)):
        raise ValueError(
            "loss_args must be a list or a dict with all optional arguments of the loss function"
        )

    perform_validation = False
    if validation_dataloader is not None:
        if not (isinstance(validation_dataloader, torch.utils.data.DataLoader)):
            raise ValueError(
                "Current implementation accept only validation data as a pytorch DataLoader"
            )
        else:
            perform_validation = True
            if validation_loss_func is None:
                validation_loss_func = loss_func
                validation_loss_args = loss_args

    if EarlyStopper is not None:
        if EarlyStopper.monitored == "validation" and not (perform_validation):
            print(
                "Early stopper monitoring is set to validation loss"
                ", but no validation data are given. "
                "Internally changing monitoring to training loss"
            )
            EarlyStopper.monitored = "train"

    loss_info = {i: [None, None] for i in range(epochs)}
    N_train = len(train_dataloader)
    N_val = 0 if validation_dataloader is None else len(validation_dataloader)
    for epoch in range(epochs):
        print(f"epoch [{epoch+1:6>}/{epochs:6>}]") if verbose else None

        train_loss = 0
        val_loss = 0
        train_loss_tot = 0
        val_loss_tot = 0

        if not (model.training):
            model.train()
        with tqdm.tqdm(
            total=N_train + N_val,
            ncols=100,
            bar_format="{desc}{percentage:3.0f}%|{bar:15}| {n_fmt}/{total_fmt}"
            " [{rate_fmt}{postfix}]",
            disable=not (verbose),
            unit=" Batch",
            file=sys.stdout,
        ) as pbar:
            for batch_idx, (X, Ytrue) in enumerate(train_dataloader):

                optimizer.zero_grad()

                # possible cases: X is tensor or not, Augmenter is iterable or not
                if isinstance(X, torch.Tensor):
                    X = X.to(device=device)
                    if augmenter is not None:
                        X = augmenter(X)
                else:
                    if augmenter is not None:
                        if isinstance(augmenter, Iterable):
                            Nmin = min(len(augmenter), len(X))
                            for i in range(Nmin):
                                X[i] = X[i].to(device=device)
                                X[i] = augmenter[i](X[i])
                            for i in range(Nmin, len(X)):
                                X[i] = X[i].to(device=device)
                        else:
                            for i in range(len(X)):
                                X[i] = X[i].to(device=device)
                                X[i] = augmenter(X[i])
                    else:
                        for i in range(len(X)):
                            X[i] = X[i].to(device=device)

                if isinstance(Ytrue, torch.Tensor):
                    if label_encoder is not None:
                        Ytrue = label_encoder(Ytrue)
                    Ytrue = Ytrue.to(device=device)
                else:
                    if label_encoder is not None:
                        if isinstance(label_encoder, Iterable):
                            Nmin = min(len(label_encoder), len(Ytrue))
                            for i in range(Nmin):
                                Ytrue[i] = label_encoder[i](Ytrue[i])
                                Ytrue[i] = Ytrue[i].to(device=device)
                            for i in range(len(Ytrue)):
                                Ytrue[i] = Ytrue[i].to(device=device)
                        else:
                            for i in range(len(Ytrue)):
                                Ytrue[i] = label_encoder(Ytrue[i])
                                Ytrue[i] = Ytrue[i].to(device=device)
                    else:
                        for i in range(len(Ytrue)):
                            Ytrue[i] = Ytrue[i].to(device=device)

                Yhat = model(X)
                train_loss = evaluate_loss(loss_func, [Yhat, Ytrue], loss_args)

                train_loss.backward()
                optimizer.step()
                train_loss_tot += train_loss.item()
                # verbose print
                if verbose:
                    pbar.set_description(f" train {batch_idx+1:8<}/{len(train_dataloader):8>}")
                    pbar.set_postfix_str(
                        f"train_loss={train_loss_tot/(batch_idx+1):.5f}, "
                        f"val_loss={val_loss_tot:.5f}"
                    )
                    pbar.update()
            train_loss_tot /= batch_idx + 1

            if lr_scheduler != None:
                lr_scheduler.step()

            # Perform validation if validation dataloader were given
            if perform_validation:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for batch_idx, (X, Ytrue) in enumerate(validation_dataloader):

                        if isinstance(X, torch.Tensor):
                            X = X.to(device=device)
                        else:
                            for i in range(len(X)):
                                X[i] = X[i].to(device=device)

                        if isinstance(Ytrue, torch.Tensor):
                            if label_encoder is not None:
                                Ytrue = label_encoder(Ytrue)
                            Ytrue = Ytrue.to(device=device)
                        else:
                            if label_encoder is not None:
                                if isinstance(label_encoder, Iterable):
                                    Nmin = min(len(label_encoder), len(Ytrue))
                                    for i in range(Nmin):
                                        Ytrue[i] = label_encoder[i](Ytrue[i])
                                        Ytrue[i] = Ytrue[i].to(device=device)
                                    for i in range(len(Ytrue)):
                                        Ytrue[i] = Ytrue[i].to(device=device)
                                else:
                                    for i in range(len(Ytrue)):
                                        Ytrue[i] = label_encoder(Ytrue[i])
                                        Ytrue[i] = Ytrue[i].to(device=device)
                            else:
                                for i in range(len(Ytrue)):
                                    Ytrue[i] = Ytrue[i].to(device=device)

                        Yhat = model(X)
                        val_loss = evaluate_loss(
                            validation_loss_func,
                            [Yhat, Ytrue],
                            validation_loss_args
                        )
                        val_loss_tot += val_loss.item()
                        if verbose:
                            pbar.set_description(
                                f"   val {batch_idx+1:8<}/{len(validation_dataloader):8>}"
                            )
                            pbar.set_postfix_str(
                                f"train_loss={train_loss_tot:.5f}, "
                                f"val_loss={val_loss_tot/(batch_idx+1):.5f}"
                            )
                            pbar.update()

                    val_loss_tot /= batch_idx + 1

        # Deal with earlystopper if given
        if EarlyStopper != None:
            updated_mdl = False
            if EarlyStopper.monitored == "validation":
                curr_monitored = val_loss_tot
            else:
                curr_monitored = train_loss_tot
            EarlyStopper.early_stop(curr_monitored)
            if EarlyStopper.record_best_weights:
                if EarlyStopper.best_loss == curr_monitored:
                    EarlyStopper.rec_best_weights(model)
                    updated_mdl = True
            if EarlyStopper():
                print(f"no improvement after {EarlyStopper.patience} epochs. Training stopped")
                if EarlyStopper.record_best_weights and not (updated_mdl):
                    EarlyStopper.restore_best_weights(model)
                if return_loss_info:
                    return loss_info
                else:
                    return

        if return_loss_info:
            loss_info[epoch] = [train_loss_tot, val_loss_tot]
    if return_loss_info:
        return loss_info

def get_performances(loader2eval, 
                     Model, 
                     device         = 'cpu', 
                     nb_classes     = 2,
                     return_scores  = True,
                     verbose        = False,
                     plot_confusion = False,
                     class_labels   = None
                    ):
    '''
    ``get_performances`` calculates numerous metrics to evaluate a Pytorch's
    model. If specified, it also display a summary and plot two confusion matrices.

    Parameters
    ----------
    loader2eval: torch.utils.data.Dataloader
        A Pytorch's Dataloader with the samples to use for the evaluation. 
    Model: torch.nn.Module
        A Pytorch's model to evaluate.
    device: torch.device, optional
        The device to use during batch forward.
        Default = 'cpu'
    nb_classes: int, optional
        The number of classes. Some operations are different between the binary
        and multiclass case.
        Default = 2
    return_scores: dict, optional
        Whether to return all the calculated metrics, predictions, and confusion
        matrices inside a dictionary.
        Default = True
    verbose: bool, optional
        Whether to print all the calculated metrics or not. A scikit-learn's
        classification report is also displayed.
        Default = False
    plot_confusion: bool, optional
        Whether to plot a confusion matrix or not.
        Default = False
    class_labels: list, optional
        A list with the labels to use for the confusion matrix plot. If None,
        values between 0 and the number of classes - 1 will be used.
        Default = None

    Returns
    -------
    scores: dict, optional
        A dictionary with a set of metrics, predictions, and confusion
        matrices calculated inside this function. The full list of values is:
            
            - 'logits': model's activations (logit output) as a numpy array.
            - 'probabilities': model's predicted probabilities as a numpy array.
            - 'predictions': model's predicted classes as a numpy array.
            - 'labels': true labels as a numpy array,
            - 'confusion': confusion matrix with absolute values as a 
              Pandas DataFrame.
            - 'confusion_normalized': normalized confusion matrix with 
              absolute values as a Pandas DataFrame.
            - 'accuracy_unbalanced': unbalanced accuracy,
            - 'accuracy_weighted': weighted accuracy,
            - 'precision_micro': micro precision,
            - 'precision_macro': macro precision,
            - 'precision_weighted': weighted precision,
            - 'precision_matrix': matrix with single class precisions,
            - 'recall_micro': micro recall,
            - 'recall_macro': macro recall,
            - 'recall_weighted': weighted recall,
            - 'recall_matrix': matrix with single calss recalls,
            - 'f1score_micro': micro f1-score,
            - 'f1score_macro': macro f1-score,
            - 'f1score_weighted': weighted f1-score,
            - 'f1score_matrix': matrix with single class f1-scores,
            - 'rocauc_micro': micro ROC AUC,
            - 'rocauc_macro': macro ROC AUC,
            - 'rocauc_weighted': weighted ROC AUC,
            - 'cohen_kappa': Cohen's Kappa score  


    '''
    # calculate logits, probabilities, and classes
    Model.to(device=device)
    Model.eval()
    ytrue = torch.zeros(len(loader2eval.dataset))
    ypred = torch.zeros_like(ytrue)
    if nb_classes<=2:
        logit = torch.zeros(len(loader2eval.dataset))
    else:
        logit = torch.zeros(len(loader2eval.dataset), nb_classes)
    proba = torch.zeros_like(logit)
    cnt=0
    for i, (X, Y) in enumerate(loader2eval):
        if isinstance(X, torch.Tensor):
            if X.device.type != device.type:
                X = X.to(device=device)
            Xshape = X.shape[0]
        else:
            for i in range(len(X)):
                if X[i].device.type != device.type:
                    X[i] = X[i].to(device=device)
            Xshape = X[0].shape[0]

        if isinstance(Y, torch.Tensor):
            ytrue[cnt:cnt+Xshape]= Y
        else:
            ytrue[cnt:cnt+Xshape]= Y[0]
        with torch.no_grad():
            yhat = Model(X)
            if isinstance(yhat, torch.Tensor):
                yhat = yhat.to(device='cpu')
            else:
                yhat = yhat[0].to(device='cpu')
                
            if nb_classes == 2:
                logit[cnt:cnt+Xshape] = torch.squeeze(yhat)
                yhat = torch.sigmoid(yhat)
                yhat = torch.squeeze(yhat)
                proba[cnt:cnt+Xshape] = yhat
                ypred[cnt:cnt+Xshape] = yhat > 0.5 
            else:
                logit[cnt:cnt+Xshape] = yhat
                yhat = torch.softmax(yhat, 1)
                proba[cnt:cnt+Xshape] = yhat
                yhat = torch.argmax(yhat, 1)
                ypred[cnt:cnt+Xshape] = torch.squeeze(yhat) 
        cnt += Xshape

    # convert to numpy for score computation
    proba = proba.numpy()
    logit = logit.numpy()
    ytrue = ytrue.numpy()
    ypred = ypred.numpy()

    # confusion matrices
    labels1 = [i for i in range(nb_classes)]
    if (class_labels is not None) and (len(class_labels)==nb_classes):
        index1  = class_labels
    else:
        index1 = [str(i) for i in range(nb_classes)]

    # get number of classes in test set to be sure that no warning will be thrown
    equal_classes_in_true = (len(np.unique(ytrue)) == nb_classes)
        
    ConfMat = confusion_matrix(ytrue, ypred, labels=labels1).T
    ConfMat_df = pd.DataFrame(ConfMat, index = index1, columns = index1)
    Acc_mat = confusion_matrix(ytrue, ypred, labels=labels1, normalize='true').T
    Acc_mat_df = pd.DataFrame(Acc_mat, index = index1, columns = index1)
    
    # accuracy, precision, recall, f1, roc_auc, cohen's kappa
    acc_unbal = accuracy_score(ytrue, ypred)

    if equal_classes_in_true:
        acc_weigh = balanced_accuracy_score(ytrue, ypred)
    else:
        acc_weigh = acc_unbal
    
    f1_mat = f1_score(ytrue, ypred, average = None, zero_division = 0.0)
    f1_micro = f1_score(ytrue, ypred, average = 'micro', zero_division = 0.0)
    f1_macro = f1_score(ytrue, ypred, average = 'macro', zero_division = 0.0)
    f1_weigh = f1_score(ytrue, ypred, average = 'weighted', zero_division = 0.0)

    prec_mat = precision_score(ytrue, ypred, average = None, zero_division=0.0)
    prec_micro = precision_score(ytrue, ypred, average = 'micro', zero_division = 0.0)
    prec_macro = precision_score(ytrue, ypred, average = 'macro', zero_division = 0.0)
    prec_weigh = precision_score(ytrue, ypred, average = 'weighted',zero_division = 0.0)

    recall_mat = recall_score(ytrue, ypred, average = None, zero_division=0.0)
    recall_micro = recall_score(ytrue, ypred, average = 'micro', zero_division = 0.0)
    recall_macro = recall_score(ytrue, ypred, average = 'macro', zero_division = 0.0)
    recall_weigh = recall_score(ytrue, ypred, average = 'weighted', zero_division = 0.0)

    if equal_classes_in_true:
        cohen_kappa = cohen_kappa_score(ytrue, ypred)
    else:
        cohen_kappa = np.nan

    
    try:
        if nb_classes == 2:
            roc_micro = roc_auc_score(ytrue, proba, average = 'micro', multi_class = 'ovo')
        else:
            roc_micro = np.nan
        roc_macro = roc_auc_score(ytrue, proba, average = 'macro', multi_class = 'ovr')
        roc_weigh = roc_auc_score(ytrue, proba, average = 'weighted', multi_class = 'ovr')
    except Exception:
        roc_micro = np.nan
        roc_macro = np.nan
        roc_weigh = np.nan

    # print everything plus a classification report if asked
    if verbose:
        print('           |-----------------------------------------|')
        print('           |                SCORE SUMMARY            |')
        print('           |-----------------------------------------|')
        print('           |  Accuracy score:                 %5.3f  |' %acc_unbal) 
        print('           |  Accuracy score weighted:        %5.3f  |' %acc_weigh) 
        print('           |-----------------------------------------|')
        print('           |  Precision score micro:          %5.3f  |' %prec_micro)
        print('           |  Precision score macro:          %5.3f  |' %prec_macro)
        print('           |  Precision score weighted:       %5.3f  |' %prec_weigh)
        print('           |-----------------------------------------|')
        print('           |  Recall score micro:             %5.3f  |' %recall_micro)
        print('           |  Recall score macro:             %5.3f  |' %recall_macro)
        print('           |  Recall score weighted:          %5.3f  |' %recall_weigh)
        print('           |-----------------------------------------|')
        print('           |  F1-score micro:                 %5.3f  |' %f1_micro)
        print('           |  F1-score macro:                 %5.3f  |' %f1_macro)
        print('           |  F1-score weighted:              %5.3f  |' %f1_weigh)
        print('           |-----------------------------------------|')
        print('           |  ROC AUC micro:                  %5.3f  |' %roc_micro)
        print('           |  ROC AUC micro:                  %5.3f  |' %roc_macro)
        print('           |  ROC AUC micro:                  %5.3f  |' %roc_weigh)
        print('           |-----------------------------------------|')
        print('           |  Cohen\'s kappa score:            %5.3f  |' %cohen_kappa)
        print('           |-----------------------------------------|')

        print(' ')
        print(classification_report(ytrue,ypred, zero_division=0))
        print(' ')

    # plot a confusion matrix if asked
    if plot_confusion:
        const_size = 30
        vmin = np.min(ConfMat)
        vmax = np.max(ConfMat)
        off_diag_mask = np.eye(*ConfMat.shape, dtype=bool)
        
        plt.figure(figsize=(14,6),layout="constrained")
        sns.set(font_scale=1.5)
        plt.subplot(1,2,1)
        sns.heatmap(ConfMat_df, vmin= 0, vmax=vmax, mask=~off_diag_mask, fmt="4d",
                    annot=True, cmap='Blues', linewidths=1, cbar_kws={'pad': 0.01},
                    annot_kws={"size": const_size / np.sqrt(len(ConfMat_df))})
        sns.heatmap(ConfMat_df, annot=True, mask=off_diag_mask, cmap='OrRd', 
                    vmin=vmin, vmax=vmax, linewidths=1, fmt="4d",
                    cbar_kws={'ticks':[], 'pad': 0.05},
                    annot_kws={"size": const_size / np.sqrt(len(ConfMat_df))})
        plt.xlabel('true labels', fontsize=20)
        plt.ylabel('predicted labels', fontsize=20)
        plt.title('Confusion Matrix', fontsize=25)
        
        sns.set(font_scale=1.5)
        plt.subplot(1,2,2)
        sns.heatmap(Acc_mat_df, vmin= -0.01, vmax=1.01, mask=~off_diag_mask, 
                    fmt=".3f", cbar_kws={'pad': 0.01},
                    annot=True, cmap='Blues', linewidths=1)
        sns.heatmap(Acc_mat_df, annot=True, mask=off_diag_mask, 
                    cmap='OrRd', fmt=".3f",
                    cbar_kws={'ticks':[], 'pad': 0.05},
                    vmin=-0.01, vmax=1.01, linewidths=1)
        plt.xlabel('true labels', fontsize=20)
        plt.ylabel('predicted labels', fontsize=20)
        plt.title('Normalized Confusion Matrix', fontsize=25)
        plt.show()

    if return_scores:
        scores = {
            'logits': logit,
            'probabilities': proba,
            'predictions': ypred,
            'labels': ytrue,
            'confusion': ConfMat_df,
            'confusion_normalized': Acc_mat_df,
            'accuracy_unbalanced': acc_unbal,
            'accuracy_weighted': acc_weigh,
            'precision_micro': prec_micro,
            'precision_macro': prec_macro,
            'precision_weighted': prec_weigh,
            'precision_matrix': prec_mat,
            'recall_micro': recall_micro,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weigh,
            'recall_matrix': recall_mat,
            'f1score_micro': f1_micro,
            'f1score_macro': f1_macro,
            'f1score_weighted': f1_weigh,
            'f1score_matrix': f1_mat,
            'rocauc_micro': roc_micro,
            'rocauc_macro': roc_macro,
            'rocauc_weighted': roc_weigh,
            'cohen_kappa': cohen_kappa    
        }
        return scores
    else:
        return
