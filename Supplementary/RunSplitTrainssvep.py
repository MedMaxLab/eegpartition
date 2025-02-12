# ===========================
#  Section 1: package import
# ===========================
# This section includes all the packages to import. 
# To run this notebook, you must install in your environment. 
# They are: numpy, pandas, matplotlib, scipy, scikit-learn, pytorch, selfeeg
import sys
sys.path.append('..')

import argparse
import glob
from itertools import chain, combinations, product
import math
import os
import random
import pickle
import copy
import warnings
warnings.filterwarnings(
    "ignore",
    message = "Using padding='same'",
    category = UserWarning
)

# IMPORT STANDARD PACKAGES
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold

# IMPORT TORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from torch.utils.data import DataLoader

# IMPORT SELFEEG 
import selfeeg
import selfeeg.models as zoo
import selfeeg.dataloading as dl

# IMPORT REPOSITORY FUNCTIONS
from AllFnc import split
from AllFnc.training import (
    lossBinary,
    lossMulti,
    train_model,
    get_performances,
    GetLearningRate
)
from AllFnc.utilities import (
    restricted_float,
    positive_float,
    positive_int_nozero,
    positive_int,
    str2bool
)

import warnings
warnings.filterwarnings(
    "ignore",
    message = "numpy.core.numeric is deprecated",
    category = DeprecationWarning
)
warnings.filterwarnings(
    "ignore",
    message = "Loaded a file with multiple EEGs",
)

def loadEEG(path: str, 
            return_label: bool=True, 
            apply_zscore: bool = True,
            onehot_label: bool = False
           ):
    with open(path, 'rb') as eegfile:
        EEG = pickle.load(eegfile)
    x = EEG['data']
    if apply_zscore:
        x = zscore(x,1)
    if return_label:
        y = EEG['labels'] - 1
        if onehot_label:
            y = F.one_hot(y, num_classes = 4)
        return x, y
    else:
        return x

class BasicDataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

if __name__ == '__main__':
    # ===========================
    #  Section 2: set parameters
    # ===========================
    # In this section all tunable parameters are instantiated. The entire training 
    # pipeline is configured here, from the task definition to the model evaluation.
    # Other code cells compute their operations using the given configuration. 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datapath",
        dest      = "dataPath",
        metavar   = "datasets path",
        type      = str,
        nargs     = 1,
        required  = True,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest      = "modelToEval",
        metavar   = "model",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'shallownet',
        choices   =['shallownet', 'deepconvnet', 'resnet', 'eegnet'],
    )
    parser.add_argument(
        "-k",
        "--kfoldstrat",
        dest      = "kfoldstrat",
        metavar   = "kfold",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'nlnso',
        choices   =['kfold', 'lnso', 'loso', 'nlnso'],
    )
    parser.add_argument(
        "-f",
        "--outer",
        dest      = "outerFold",
        metavar   = "outer fold",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 1,
    )
    parser.add_argument(
        "-i",
        "--inner",
        dest      = "innerFold",
        metavar   = "inner fold",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 1,
    )
    parser.add_argument(
        "-z",
        "--zscore",
        dest      = "z_score", 
        metavar   = "zscore",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
    )
    parser.add_argument(
        "-b",
        "--batch",
        dest      = "batchsize",
        metavar   = "batch size",
        type      = positive_int_nozero,
        nargs     = '?',
        required  = False,
        default   = 64,
    )
    parser.add_argument(
        "-l",
        "--learningrate",
        dest      = "lr",
        metavar   = "learning rate",
        type      = positive_float,
        nargs     = '?',
        required  = False,
        default   = 0.0,
    )
    parser.add_argument(
        "-g",
        "--gpu",
        dest      = "gpudevice",
        metavar   = "torch device",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'cuda:0',
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest      = "verbose",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
    )
    args = vars(parser.parse_args())
    
    if args['verbose']:
        print('running training with the following parameters:')
        print(' ')
        for key in args:
            if key == 'dataPath':
                print( f"{key:15} ==> {args[key][0]:<15}") 
            else:
                print( f"{key:15} ==> {args[key]:<15}") 
    
    dataPath       = args['dataPath'][0].casefold()
    modelToEval    = args['modelToEval'].casefold()
    kfoldstrat     = args['kfoldstrat'].casefold()
    outerFold      = args['outerFold'] - 1
    innerFold      = args['innerFold'] - 1
    z_score        = args['z_score']
    batchsize      = args['batchsize']
    verbose        = args['verbose']
    lr             = args['lr']
    device         = torch.device(args['gpudevice'])

    seed = 83136297
    
    # ==================================
    #  Section 3: create partition list
    # ==================================
    
    Nsubj = 54
    if kfoldstrat in ['lnso', 'nlnso']:
        
        Nouter = 10
        Ninner = 10
        foldToEval = outerFold*Ninner + innerFold
        partition_list = split.create_nested_kfold_subject_split(Nsubj, Nouter, Ninner)
        train_id   = partition_list[foldToEval][0]
        val_id     = partition_list[foldToEval][1]
        test_id    = partition_list[foldToEval][2]
        if kfoldstrat == 'lnso':
            train_id += val_id
            train_id.sort()
            val_id = test_id
            test_id = []

    elif kfoldstrat == 'loso':
        
        Nouter = Nsubj
        Ninner = Nsubj-1
        foldToEval = outerFold*Ninner + innerFold
        partition_list = split.create_nested_kfold_subject_split(Nsubj, Nouter, Ninner) 
        train_id   = partition_list[foldToEval][0]
        val_id     = partition_list[foldToEval][1]
        test_id    = partition_list[foldToEval][2]
        if kfoldstrat == 'loso':
            train_id += val_id
            train_id.sort()
            val_id = test_id
            test_id = []   
    
    else:
        foldToEval = outerFold

    # ======================================
    # Section 4: set the training parameters
    # =====================================
    Chan           = 56
    freq           = 125
    nb_classes     = 4
    Samples        = 500
    datasetID      = '73'
    overlap        = 0.0
    workers        = 0
    window         = 4.0
    exclude_id     = None
    pipelineToEval = 'SSVEP'
    taskToEval     = 'ssvep'
    class_labels   = ['up', 'left', 'right', 'down']
    eegpath        = dataPath + pipelineToEval
        
    
    # =====================================================
    #  Section 5: Define pytorch's Datasets and dataloaders
    # =====================================================
    
    # Now that everything is ready, let's define all the Datasets and Dataloaders. 
    # The dataset is defined by using the selfEEG EEGDataset custom class, 
    # which includes an option to preload the entire dataset.
    
    # GetEEGPartitionNumber doesn't need the labels
    loadEEG_args = {
        'return_label': False,
        'apply_zscore': z_score
    }
    
    glob_input = [datasetID + '_*.pickle']
    
    # calculate dataset length with selfgeeg. Basically it automatically
    # retrieves all the partitions that can be extracted from each EEG signal
    EEGlen = dl.get_eeg_partition_number(
        eegpath,
        freq,
        window,
        overlap, 
        file_format = glob_input,
        load_function = loadEEG,
        optional_load_fun_args = loadEEG_args,
        includePartial = False,
        verbose = verbose
    )
    
    # Now we also need to load the labels
    loadEEG_args['return_label'] = True
    
    # Set functions to retrieve dataset, subject, and session from each filename.
    # They will be used by GetEEGSplitTable to perform a subject based split
    dataset_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[0])
    subject_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[1]) 
    session_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[2])
    
    if kfoldstrat in ['lnso', 'loso', 'nlnso']:
        # Now call the GetEEGSplitTable. Since Parkinson task merges two datasets
        # we need to differentiate between this and other tasks
        EEGsplit= dl.get_eeg_split_table(
            partition_table      = EEGlen,
            val_data_id          = val_id,
            test_data_id         = test_id,
            exclude_data_id      = exclude_id,
            split_tolerance      = 0.001,
            dataset_id_extractor = dataset_id_ex if taskToEval=='parkinson' else subject_id_ex,
            subject_id_extractor = subject_id_ex if taskToEval=='parkinson' else session_id_ex,
            perseverance         = 10000,
            seed                 = seed
        )
        
        if verbose:
            print(' ')
            print('Subjects used for validation: ', val_id)
            if kfoldstrat in ['nlnso', 'nloso', 'ploso']:
                print('Subjects used for test: ', test_id)
        
        # Define Datasets and preload all data
        trainset = dl.EEGDataset(
            EEGlen, EEGsplit, [freq, window, overlap], 'train', 
            supervised             = True, 
            label_on_load          = True,
            multilabel_on_load     = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args
        )
        trainset.preload_dataset()
    
        # lnso and loso doesn't have a test set but,
        valset = dl.EEGDataset(
            EEGlen, EEGsplit, [freq, window, overlap], 'validation',
            supervised             = True, 
            label_on_load          = True,
            multilabel_on_load     = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args
        )
        valset.preload_dataset()
    
        testset = dl.EEGDataset(
            EEGlen, EEGsplit, [freq, window, overlap],
            'test' if kfoldstrat in ['nlnso', 'nloso', 'ploso'] else 'validation',
            supervised             = True,
            label_on_load          = True,
            multilabel_on_load     = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args
        )
        testset.preload_dataset()
        
        trainset.y_preload = trainset.y_preload.to(dtype = torch.long)
        valset.y_preload   = valset.y_preload.to(dtype = torch.long)
        testset.y_preload  = testset.y_preload.to(dtype = torch.long)        
        trainset.x_preload = trainset.x_preload.to(device=device)
        trainset.y_preload = trainset.y_preload.to(device=device)
        valset.x_preload = valset.x_preload.to(device=device)
        valset.y_preload = valset.y_preload.to(device=device)
        testset.x_preload = testset.x_preload.to(device=device)
        testset.y_preload = testset.y_preload.to(device=device)

    # same things but for sample based approaches
    else:
    
        EEGsplit= dl.get_eeg_split_table(
            partition_table      = EEGlen,
            exclude_data_id      = exclude_id,
            split_tolerance      = 0.001,
            dataset_id_extractor = subject_id_ex,
            subject_id_extractor = session_id_ex,
            perseverance         = 10000,
            seed                 = seed
        )
        EEGsplit.loc[EEGsplit['split_set']!=-1, 'split_set'] = 0
        
        # Define Datasets and preload all data
        dataset = dl.EEGDataset(
            EEGlen, EEGsplit, [freq, window, overlap], 'train', 
            supervised             = True, 
            label_on_load          = True,
            load_function          = loadEEG,
            optional_load_fun_args = loadEEG_args,
            multilabel_on_load = True
        )
        dataset.preload_dataset()
        dataset.y_preload = dataset.y_preload.to(dtype = torch.long)

        # use Stratified K-fold in sklearn to get 
        skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        ydummy = dataset.y_preload.numpy()
        Xdummy = np.arange(dataset.x_preload.shape[0])
        for i, (train_index, test_index) in enumerate(skf.split(Xdummy, ydummy)):
            if i == foldToEval:
                break

        Xtrain = dataset.x_preload[train_index]
        Ytrain = dataset.y_preload[train_index]
        Xval   = dataset.x_preload[test_index]
        Yval   = dataset.y_preload[test_index]
        Xtest  = dataset.x_preload[test_index]
        Ytest  = dataset.y_preload[test_index]
        
        Xtrain = Xtrain.to(device=device)
        Ytrain = Ytrain.to(device=device)
        Xval   = Xval.to(device=device)
        Yval   = Yval.to(device=device)
        Xtest  = Xtest.to(device=device)
        Ytest  = Ytest.to(device=device)
        
        trainset = BasicDataset(Xtrain, Ytrain)
        valset   = BasicDataset(Xval, Yval)
        testset  = BasicDataset(Xtest, Ytest)
    
    # ---- END OF BIG IF FOR DATA PARTITION ----
    
    # Finally, Define Dataloaders
    # (no need to use more workers in validation and test dataloaders)
    trainloader = DataLoader(dataset = trainset, batch_size = batchsize,
                             shuffle = True, num_workers = workers)
    valloader = DataLoader(dataset = valset, batch_size = batchsize,
                           shuffle = False, num_workers = 0)
    testloader = DataLoader(dataset = testset, batch_size = batchsize,
                            shuffle = False, num_workers = 0)
    
    # ===================================================
    #  Section 6: define the loss, model, and optimizer
    # ==================================================
    
    lossVal = None
    validation_loss_args = []
    lossFnc = lossMulti

    # SET SEEDS FOR REPRODUCIBILITY
    # why this seed? It's MedMax in ASCII!
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    
    # define model
    if modelToEval.casefold() == 'shallownet':
        Mdl = zoo.ShallowNet(nb_classes, Chan, Samples)
    elif modelToEval.casefold() == 'deepconvnet':
        Mdl = zoo.DeepConvNet(
            nb_classes, Chan, Samples, kernLength = 10, F = 25, Pool = 3,
            stride = 3, batch_momentum = 0.1, dropRate = 0.5,
            max_norm = None, max_dense_norm = None
        )
    elif modelToEval.casefold() == 'eegnet':
        Mdl = zoo.EEGNet(
            nb_classes, Chan, Samples, depthwise_max_norm=None, norm_rate=None
        )
    elif modelToEval.casefold() == 'resnet':
        Mdl = zoo.ResNet1D(
            nb_classes, Chan, Samples, Layers = [3, 4, 6, 3],
            inplane = 16, kernLength = 7, addConnection = False
        )
    
    MdlBase = copy.deepcopy(Mdl)
    Mdl.to(device = device)
    Mdl.train()
    if verbose:
        print(' ')
        ParamTab = selfeeg.utils.count_parameters(Mdl, False, True, True)
        print(' ')
    
    
    if lr == 0:
        lr = GetLearningRate(modelToEval, taskToEval)
        if verbose:
            print(' ')
            print('used learning rate', lr)
    gamma = 0.995
    optimizer = torch.optim.Adam(Mdl.parameters(), lr = lr)
    #optimizer = torch.optim.Adam(Mdl.parameters(), lr = lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
    
    # Define selfEEG's EarlyStopper with large patience to act as a model checkpoint
    earlystop = selfeeg.ssl.EarlyStopping(
        patience = 15, 
        min_delta = 1e-04, 
        record_best_weights = True
    )
    
    # =============================
    #  Section 7: train the model
    # =============================
    loss_summary=train_model(
        model                 = Mdl,
        train_dataloader      = trainloader,
        epochs                = 100,
        optimizer             = optimizer,
        loss_func             = lossFnc, 
        lr_scheduler          = scheduler,
        EarlyStopper          = earlystop,
        validation_dataloader = valloader,
        validation_loss_func  = lossVal,
        validation_loss_args  = validation_loss_args,
        verbose               = verbose,
        device                = device,
        return_loss_info      = True
    )
    
    
    # ===============================
    #  Section 8: evaluate the model
    # ===============================
    earlystop.restore_best_weights(Mdl)
    Mdl.to(device=device)
    Mdl.eval()
    scores = get_performances(
        loader2eval    = testloader, 
        Model          = Mdl, 
        device         = device,
        nb_classes     = nb_classes,
        return_scores  = True,
        verbose        = verbose,
        plot_confusion = False,
        class_labels   = class_labels
    )

    if kfoldstrat in ['nlnso']:
        scores['validation_scores'] = get_performances(
            loader2eval    = valloader, 
            Model          = Mdl, 
            device         = device,
            nb_classes     = nb_classes,
            return_scores  = True,
            verbose        = verbose,
            plot_confusion = False,
            class_labels   = class_labels
        )
    
    
    # ==================================
    #  Section 9: Save model and metrics
    # ==================================
    
    # Set the output path
    start_piece_mdl = '../SSVEPClassification/Models/'
    start_piece_res = '../SSVEPClassification/Results/'
    task_piece = 'svp'
    
    if modelToEval.casefold() == 'shallownet':
        mdl_piece = 'shn'
    elif modelToEval.casefold() == 'deepconvnet':
        mdl_piece = 'dcn'
    elif modelToEval.casefold() == 'eegnet':
        mdl_piece = 'egn'
    elif modelToEval.casefold() == 'resnet':
        mdl_piece = 'res'

    
    pipe_piece = 'flt' 
    freq_piece = '125'

    if kfoldstrat =='lnso':
        start_piece_mdl = start_piece_mdl + 'LNSO' + os.sep
        start_piece_res = start_piece_res + 'LNSO' + os.sep
        cv_piece = 'slnso'
    elif kfoldstrat =='loso':
        start_piece_mdl = start_piece_mdl + 'LOSO' + os.sep
        start_piece_res = start_piece_res + 'LOSO' + os.sep
        cv_piece = 'sloso'
    elif kfoldstrat =='nlnso':
        start_piece_mdl = start_piece_mdl + 'NLNSO' + os.sep
        start_piece_res = start_piece_res + 'NLNSO' + os.sep
        cv_piece = kfoldstrat
    elif kfoldstrat =='kfold':
        start_piece_mdl = start_piece_mdl + 'KFOLD' + os.sep
        start_piece_res = start_piece_res + 'KFOLD' + os.sep
        cv_piece = kfoldstrat

    
    out_piece = str(outerFold+1).zfill(3)
    in_piece = str(innerFold+1).zfill(3)
    lr_piece = str(int(lr*1e6)).zfill(6)
    chan_piece = str(Chan).zfill(3)
    win_piece = str(round(window)).zfill(3)
    
    file_name = '_'.join(
        [cv_piece, task_piece, pipe_piece, freq_piece, mdl_piece, 
         out_piece, in_piece, lr_piece, chan_piece, win_piece
        ]
    )
    model_path = start_piece_mdl + file_name + '.pt'
    results_path = start_piece_res + file_name + '.pickle'
    
    if verbose:
        print('saving model and results in the following paths')
        print(model_path)
        print(results_path)
    
    # Save the model
    #Mdl.eval()
    #Mdl.to(device='cpu')
    #torch.save(Mdl.state_dict(), model_path)
    
    # Save the scores
    with open(results_path, 'wb') as handle:
        pickle.dump(scores, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    if verbose:
        print('run complete')
