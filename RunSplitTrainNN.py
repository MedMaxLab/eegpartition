# ===========================
#  Section 1: package import
# ===========================
# This section includes all the packages to import. 
# To run this notebook, you must install in your environment. 
# They are: numpy, pandas, matplotlib, scipy, scikit-learn, pytorch, selfeeg

import argparse
import glob
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

# IMPORT TORCH
import torch
import torch.nn as nn
from torchaudio import transforms
from torch.utils.data import DataLoader

# IMPORT SELFEEG 
import selfeeg
import selfeeg.models as zoo
import selfeeg.dataloading as dl

# IMPORT REPOSITORY FUNCTIONS
from AllFnc import split
from AllFnc.training import (
    loadEEG,
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
    
    help_d = """
    RunSplitTrainNN run a single training with a specific split 
    extracted from a chosen cross validation modality. Many parameters 
    to set up the training process can be set. All of them will be recorded
    using a unique file name.
    The required input is the root dataset path.
    Others have a default in case you want to check a single demo run.
    
    Example:
    
    $ python RunSplitTrainNN -d /path/to/data
    
    """
    parser = argparse.ArgumentParser(description=help_d)
    parser.add_argument(
        "-D",
        "--datapath",
        dest      = "dataPath",
        metavar   = "datasets path",
        type      = str,
        nargs     = 1,
        required  = True,
        help      = """
        The dataset path. This is expected to be static across all trainings. 
        dataPath must point to a directory which contains four subdirecotries, one with 
        all the pickle files containing EEGs preprocessed with a specific pipeline.
        Subdirectoties are expected to have the following names, which are the same as
        the preprocessing pipelinea to evaluate: 1) raw; 2) filt; 3) ica; 4) icasr
        """,
    )
    parser.add_argument(
        "-p",
        "--pipeline",
        dest      = "pipelineToEval",
        metavar   = "preprocessing pipeline",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'filt',
        choices   =['raw', 'filt', 'ica', 'icasr'],
        help      = """
        The pipeline to consider. It can be one of the following:
        1) raw; 2) filt; 3) ica; 4) icasr
        """,
    )
    parser.add_argument(
        "-t",
        "--task",
        dest      = "taskToEval",
        metavar   = "task",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'alzheimerca',
        choices   =['alzheimer', 'parkinson', 'bci'],
        help      = """
        The task to evaluate. It can be one of the following:
        1) alzheimer; 2) parkinson; 3) bci;
        """,
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
        choices   =['shallownet', 'eegnet', 'deepconvnet', 'resnet'],
        help      = """
        The model to evaluate. It can be one of the following:
        1) shallownet; 2) eegnet; 3) deepconvnet; 4) resnet;
        """,
    )
    parser.add_argument(
        "-k",
        "--kfoldstrat",
        dest      = "kfoldstrat",
        metavar   = "kfold",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = 'nnlns',
        choices   =['nnlns'],
        help      = """
        The cross validation strategy to evaluate. It can be one of the following:
        1) nnlns (nested nested leave-n-subject-out);
        """,
    )
    parser.add_argument(
        "-R",
        "--repetition",
        dest      = "repetitionFold",
        metavar   = "repetition fold",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 1,
        help      = """
        The repetition fold to evaluate.
        This fold basically determines which subjects will be left out before
        actually running a N-NLSO.
        So, each dataset is partitioned in this way:
            1) dataset --> Bias partition, N-LNSO Partition
            2) N-LNSO partition  --> Train, Validation, Test Set
        """
    )
    parser.add_argument(
        "-O",
        "--outer",
        dest      = "outerFold",
        metavar   = "outer fold",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 1,
        help      = """
        The outer fold to evaluate.
        It will be used for every possible CV strategy.
        For the N-NLSO Cross validation technique, an inner fold must also be given.
        """
    )
    parser.add_argument(
        "-I",
        "--inner",
        dest      = "innerFold",
        metavar   = "inner fold",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 1,
        help      = """
        The inner fold to evaluate.
        It will be used only for the N-NLSO Cross validation technique.
        """
    )
    parser.add_argument(
        "-d",
        "--downsample",
        dest      = "downsample",
        metavar   = "downsample",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
        help      = """
        A boolean that set if downsampling at 125 Hz should be applied or not.
        If set to False, data at 250 Hz (160 Hz for the MI task) will be used.
        """
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
        help      = """
        A boolean that set if the z-score should be applied or not. 
        The presented analysis applied the z-score.
        """
    )
    parser.add_argument(
        "-r",
        "--rminterp",
        dest      = "rem_interp",
        metavar   = "remove interpolated",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = True,
        help      = """
        A boolean that set if the interpolated channels should be 
        removed or not. BIDSAlign aligns all EEGs to a common 61 channel
        template based on the 10_10 International System.
        """
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
        help      = """
        Define the Batch size. It is suggested to use 64 or 128.
        The experimental analysis was performed on batch 64.
        """
    )
    parser.add_argument(
        "-o",
        "--overlap",
        dest      = "overlap",
        metavar   = "windows overlap",
        type      = restricted_float,
        nargs     = '?',
        required  = False,
        default   = 0.0,
        help      = """
        The overlap between time windows. Higher values means more samples 
        but higher correlation between them. 0.25 is a good trade-off.
        Must be a value in [0,1). Here, no overlap is used.
        """
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
        help      = """
        The learning rate. If left to its default (zero) a proper learning rate
        will be chosen depending on the model and task to evaluate. Optimal learning
        rates were identified by running multiple trainings with different set of values.
        Must be a positive value
        """
    )
    parser.add_argument(
        "-w",
        "--window",
        dest      = "window",
        metavar   = "window",
        type      = positive_float,
        nargs     = '?',
        required  = False,
        default   = 4.0,
        help      = """
        The window (input) size, in seconds. Each EEG will be partitioned in
        windows of length equals to the one specified by this input.
        c was the first available letter.
        """
    )
    parser.add_argument(
        "-W",
        "--workers",
        dest      = "workers",
        metavar   = "dataloader workers",
        type      = positive_int,
        nargs     = '?',
        required  = False,
        default   = 0,
        help      = """
        The number of workers to set for the dataloader. Datasets are preloaded
        for faster computation, so 0 is preferred due to known issues on values
        greater than 1 for some os, and to not increase too much the memory usage.
        """
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
        help      = "The cuda device to use.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest      = "verbose",
        type      = str2bool,
        nargs     = '?',
        required  = False,
        default   = False,
        help      = """
        Set the verbosity level of the whole script. If True, information about
        the choosen split, and the training progression will be displayed
        """
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
    pipelineToEval = args['pipelineToEval'].casefold()
    taskToEval     = args['taskToEval'].casefold()
    modelToEval    = args['modelToEval'].casefold()
    kfoldstrat     = args['kfoldstrat'].casefold()
    outerFold      = args['outerFold'] - 1
    innerFold      = args['innerFold'] - 1
    repFold        = args['repetitionFold'] - 1
    downsample     = args['downsample']
    z_score        = args['z_score']
    rem_interp     = args['rem_interp']
    batchsize      = args['batchsize']
    overlap        = args['overlap']
    workers        = args['workers']
    window         = args['window']
    verbose        = args['verbose']
    lr             = args['lr']
    device         = torch.device(args['gpudevice'])

    seed = 83136297
    
    if taskToEval == 'bci' and window>4.1:
        raise ValueError(
            'bci task has trials of length 4.1 seconds. Cannot exceed this number'
        )
    elif taskToEval == 'bci' and window == 4.0:
        window = 4.1
    
    # ==================================
    #  Section 3: create partition list
    # ==================================
    
    # fold to eval is the correct index to get the desired train/val/test partition
    # in case of a N-LNSO CV strat
    if taskToEval == 'alzheimer':
        Nsubj = 88
    elif taskToEval == 'parkinson':
        Nsubj = 81
    elif taskToEval == 'bci':
        Nsubj = 106
    
    
        
    Nouter = 10
    Ninner = 10
    Nbias  = 10

    # to run a lnso, just fix inner to 1 
    foldToEval = outerFold*Ninner + innerFold
    
    if taskToEval == 'alzheimer':
        a_id = [i for i in range(1,37)]  # ALZ = subjects 1 to 36;
        c_id = [i for i in range(37,66)] # CTL = subjects 37 to 65;
        f_id = [i for i in range(66,89)] # FTD = subjects 66 to 88;
        
        part_a = split.create_nested_kfold_subject_split(a_id, Nbias, 1)
        part_c = split.create_nested_kfold_subject_split(c_id, Nbias, 1)
        part_f = split.create_nested_kfold_subject_split(f_id, Nbias, 1)
        partition_1 = split.merge_partition_lists(part_a, part_c, Nbias, 1)
        partition_list = split.merge_partition_lists(partition_1, part_f, Nbias, 1)
        
        bias_id = partition_list[repFold][2]
        for i in bias_id:
            try:
                a_id.remove(i)
            except ValueError:
                try:
                    c_id.remove(i)
                except ValueError:
                    f_id.remove(i)
        
        part_a = split.create_nested_kfold_subject_split(a_id, Nouter, Ninner)
        part_c = split.create_nested_kfold_subject_split(c_id, Nouter, Ninner)
        part_f = split.create_nested_kfold_subject_split(f_id, Nouter, Ninner)
        
        partition_1 = split.merge_partition_lists(part_a, part_c, Nouter, Ninner)
        partition_list = split.merge_partition_lists(partition_1, part_f, Nouter, Ninner)
    
    elif taskToEval == 'parkinson':
        # In this case, two datasets were merged to increase the number 
        # of subjects. So, there are two partition lists to create
    
        #ds003490 - ID 5 - 3Stim
        ctl_id = [i for i in range(28,51)] + [3,5]
        pds_id = [i for i in range(6,28)] + [1,2,4]
        part_c = split.create_nested_kfold_subject_split(ctl_id, Nbias, 1)
        part_p = split.create_nested_kfold_subject_split(pds_id, Nbias, 1)
        partition_list_1 = split.merge_partition_lists(part_c, part_p, Nbias, 1)
        
        bias_id_1 = partition_list_1[repFold][2]
        for i in bias_id_1:
            try:
                ctl_id.remove(i)
            except ValueError:
                pds_id.remove(i)
        
        part_c = split.create_nested_kfold_subject_split(ctl_id, Nouter, Ninner)
        part_p = split.create_nested_kfold_subject_split(pds_id, Nouter, Ninner)
        partition_list_1 = split.merge_partition_lists(part_c, part_p, Nouter, Ninner)
        
        #ds002778 - ID 8 - UCSD
        part_c = split.create_nested_kfold_subject_split([i for i in range(1,17)], Nbias, 1)
        part_p = split.create_nested_kfold_subject_split([i for i in range(17,32)], Nbias, 1)
        partition_list_2 = split.merge_partition_lists(part_c, part_p, Nbias, 1)
        
        ctl_id = [i for i in range(1,17)]
        pds_id = [i for i in range(17,32)]
        
        bias_id_2 = partition_list_2[repFold][2]
        for i in bias_id_2:
            try:
                ctl_id.remove(i)
            except ValueError:
                pds_id.remove(i)
        
        part_c = split.create_nested_kfold_subject_split(ctl_id, Nouter, Ninner)
        part_p = split.create_nested_kfold_subject_split(pds_id, Nouter, Ninner)
        partition_list_2 = split.merge_partition_lists(part_c, part_p, Nouter, Ninner)
        
        bias_id = {5: bias_id_1, 8: bias_id_2}
        
    else:
        # three subjects were excluded for the known issue of having 
        # a sampling rate of 128 Hz instead of 160 Hz (and strange trial length).
        subject_list = [i for i in range(1,110) if i not in [88,92,100]] 
        partition_list = split.create_nested_kfold_subject_split(subject_list, Nbias, 1)
        bias_id = partition_list[repFold][2]
        for i in bias_id:
            subject_list.remove(i)
        partition_list = split.create_nested_kfold_subject_split(subject_list, Nouter, Ninner)

    # remember that a not nested subject-out cross validation techniques
    # doesn't have a validation set so:
    # 1. we merge train and validation (inner fold) into a unique train set
    # 2. we use the here called test set (outer fold) as a validation
    if taskToEval == 'parkinson':
        train_id   = { 5: partition_list_1[foldToEval][0], 
                       8: partition_list_2[foldToEval][0]}
        val_id     = { 5: partition_list_1[foldToEval][1], 
                       8: partition_list_2[foldToEval][1]}
        test_id    = { 5: partition_list_1[foldToEval][2],
                       8: partition_list_2[foldToEval][2]}
    else:
        train_id   = partition_list[foldToEval][0]
        val_id     = partition_list[foldToEval][1]
        test_id    = partition_list[foldToEval][2]

    if verbose:
        print("\n repetition IDs: \n", bias_id)
        print("\n training IDs: \n",train_id)
        print("\n validation IDs: \n", val_id)
        print("\n test IDs: \n", test_id)
    # ======================================
    # Section 4: set the training parameters
    # =====================================
    
    # This section sets other parameters necessary to start the training pipeline. 
    # Such parameters are necessary to:
    # customize the EEG loading function.
    # define the Pytorch's Dataset and Dataloader classes.
    # define the NN models.
    
    # Define the Path to EEG data as a concatenation of:
    # 1) the root path
    # 2) the preprocessing pipeline
    if dataPath[-1] != os.sep:
        dataPath += os.sep
    if pipelineToEval[-1] != os.sep:
        eegpath = dataPath + pipelineToEval + os.sep
    else:
        eegpath = dataPath + pipelineToEval
    
    # Define the number of Channels to use. 
    # Basically 61 due to BIDSAlign channel system alignment.
    # Note that BIDSAlign DOES NOT delete any original channel by default.
    if rem_interp:
        if taskToEval == 'alzheimer':
            Chan = 19
        elif taskToEval == 'parkinson':
            Chan = 32
        elif taskToEval == 'bci':
            Chan = 61
    else:
        Chan = 61
    
    # Define the sampling rate. 125 or 250 depending on the downsample option
    # NOTE: that motorimagery has an original sampling rate of 160.Its downsample
    #       will be handeled during the creation of the dataset and dataloader
    if taskToEval == 'bci':
        freq = 160
    else:
        freq = 125 if downsample else 250
    
    # Define the number of classes to predict.
    if taskToEval == 'alzheimer':
        nb_classes = 3
    elif taskToEval == 'bci':
        nb_classes = 4
    else:
        nb_classes = 2
    
    # For selfEEG's models instantiation
    Samples = int(freq*window)
    
    # Set the Dataset ID for glob.glob operation in SelfEEG's GetEEGPartitionNumber().
    # It is a single number for every task except for PD that merges two datasets
    if taskToEval == 'alzheimer':
        datasetID = '10'
    elif taskToEval == 'bci':
        datasetID = '99'
    else:
        datasetID_1 = '5' # EEG 3-Stim
        datasetID_2 = '8' # UC SD
    
    # Set the class label in case of plot of functions
    if taskToEval == 'alzheimer':
        class_labels = ['CTL', 'FTD', 'AD']
    elif taskToEval == 'bci':
        class_labels = ['LeftMove', 'RightMove', 'LeftImg', 'RightImg']
    else:
        class_labels = ['CTL', 'PD']
        
    
    # =====================================================
    #  Section 5: Define pytorch's Datasets and dataloaders
    # =====================================================
    
    # Now that everything is ready, let's define all the Datasets and Dataloaders. 
    # The dataset is defined by using the selfEEG EEGDataset custom class, 
    # which includes an option to preload the entire dataset.
    
    # GetEEGPartitionNumber doesn't need the labels
    loadEEG_args = {
        'return_label': False, 
        'downsample': downsample, 
        'use_only_original': rem_interp,
        'apply_zscore': z_score
    }
    
    if taskToEval.casefold() == 'parkinson':
        glob_input = [datasetID_1 + '_*.pickle', datasetID_2 + '_*.pickle' ]
    else:
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
        includePartial = False if overlap == 0 else True,
        verbose = verbose
    )
    
    # Now we also need to load the labels
    loadEEG_args['return_label'] = True
    
    # Set functions to retrieve dataset, subject, and session from each filename.
    # They will be used by GetEEGSplitTable to perform a subject based split
    dataset_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[0])
    subject_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[1]) 
    session_id_ex  = lambda x: int(x.split(os.sep)[-1].split('_')[2]) 
    
    # Set subjects to exclude. This apply for binary version of multilabel datasets
    if taskToEval == 'bci':
        exclude_id = [88,92,100]
    else:
        exclude_id = None
    
    
    # This block is pretty big but really important
    # It creates the training, validation, test (test only if needed) sets
    # based on the partition strategy
    # Basically:
    # 1) if the validation procedure is of the "subject-out" type, do something
    # 2) if the validation procedure is a standard K-fold, do another thing
    #
    # NOTE THAT, aside for the nlnso and the nloso strats, test sets aren't needed
    # but they are created as a copy of the validation sets just to better
    # organize the code. Just take a note of this comment before assuming
    # that we implemented something wrong
    #
    # Now call the GetEEGSplitTable. Since Parkinson task merges two datasets
    # we need to differentiate between this and other tasks
    EEGsplit= dl.get_eeg_split_table(
        partition_table      = EEGlen,
        val_data_id          = val_id,
        test_data_id         = test_id,
        exclude_data_id      = bias_id if exclude_id is None else bias_id + exclude_id,
        split_tolerance      = 0.001,
        dataset_id_extractor = dataset_id_ex if taskToEval=='parkinson' else subject_id_ex,
        subject_id_extractor = subject_id_ex if taskToEval=='parkinson' else session_id_ex,
        perseverance         = 10000,
        seed                 = seed #not needed since we are giving subject IDs
    )
    
    # Define Datasets and preload all data
    trainset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'train', 
        supervised             = True, 
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args
    )
    trainset.preload_dataset()

    # lnso and loso doesn't have a test set but,
    valset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'validation',
        supervised             = True, 
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args
    )
    valset.preload_dataset()

    testset = dl.EEGDataset(
        EEGlen, EEGsplit, [freq, window, overlap], 'test',
        supervised             = True,
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args
    )
    testset.preload_dataset()

    EEGsplit2 = copy.deepcopy(EEGsplit)
    EEGsplit2.loc[EEGsplit2['split_set']>=0, 'split_set'] = 1
    EEGsplit2.loc[EEGsplit2['split_set']==-1, 'split_set'] = 2
    biasset = dl.EEGDataset(
        EEGlen, EEGsplit2, [freq, window, overlap], 'test',
        supervised             = True,
        label_on_load          = True,
        load_function          = loadEEG,
        optional_load_fun_args = loadEEG_args
    )
    biasset.preload_dataset()

    # Convert to long if task is multiclass classification.
    # This avoids Value Errors during cross entropy loss calculation
    if (taskToEval == 'alzheimer') or (taskToEval == 'bci'):
        trainset.y_preload = trainset.y_preload.to(dtype = torch.long)
        valset.y_preload   = valset.y_preload.to(dtype = torch.long)
        testset.y_preload  = testset.y_preload.to(dtype = torch.long)
        biasset.y_preload  = biasset.y_preload.to(dtype = torch.long)
    
    # Apply special downsampling to preloaded dataset if task is motorimagery
    # We will use torchaudio Resample function, which is really good
    if taskToEval == 'bci' and downsample:
        tr = transforms.Resample(160, 125, 'sinc_interp_hann', 48)
        trainset.x_preload = tr(trainset.x_preload)
        valset.x_preload   = tr(valset.x_preload)
        testset.x_preload  = tr(testset.x_preload)
        biasset.x_preload  = tr(biasset.x_preload)
        Samples, freq, window = 513, 125, 513/125
    
    trainset.x_preload = trainset.x_preload.to(device=device)
    trainset.y_preload = trainset.y_preload.to(device=device)
    valset.x_preload = valset.x_preload.to(device=device)
    valset.y_preload = valset.y_preload.to(device=device)
    testset.x_preload = testset.x_preload.to(device=device)
    testset.y_preload = testset.y_preload.to(device=device)
    biasset.x_preload = biasset.x_preload.to(device=device)
    biasset.y_preload = biasset.y_preload.to(device=device)
    
    if verbose:
        # plot split statistics
        labels = np.zeros(len(EEGlen))
        for i in range(len(EEGlen)):
            path = EEGlen.iloc[i,0]
            with open(path, 'rb') as eegfile:
                EEG = pickle.load(eegfile)
            labels[i] = EEG['label']
        print("train - val - test split ratios:")
        dl.check_split(EEGlen, EEGsplit, labels)
        print("left out subject ratios (check test set ratios):")
        dl.check_split(EEGlen, EEGsplit2, labels)
    
    # ---- END OF BIG IF FOR DATA PARTITION ----
    
    # Finally, Define Dataloaders
    # (no need to use more workers in validation and test dataloaders)
    trainloader = DataLoader(dataset = trainset, batch_size = batchsize,
                             shuffle = True, num_workers = workers)
    valloader = DataLoader(dataset = valset, batch_size = batchsize,
                           shuffle = False, num_workers = 0)
    testloader = DataLoader(dataset = testset, batch_size = batchsize,
                            shuffle = False, num_workers = 0)
    biasloader = DataLoader(dataset = biasset, batch_size = batchsize,
                            shuffle = False, num_workers = 0)
    if verbose:
        print("datasets' length \n")
        print("\n repetition: \n", len(biasset))
        print("\n training: \n",len(trainset))
        print("\n validation: \n", len(valset))
        print("\n test: \n", len(testset))
    
    # ===================================================
    #  Section 6: define the loss, model, and optimizer
    # ==================================================
    
    lossVal = None
    validation_loss_args = []
    if (taskToEval == "alzheimer") or (taskToEval == 'bci'):
        lossFnc = lossMulti
    else:
        lossFnc = lossBinary
    
    # SET SEEDS FOR REPRODUCIBILITY
    # why this seed? It's MedMax in ASCII!
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    
    # define model
    if modelToEval.casefold() == 'shallownet':
        Mdl = zoo.ShallowNet(nb_classes, Chan, Samples)
    elif modelToEval.casefold() == 'eegnet':
        Mdl = zoo.EEGNet(
            nb_classes, Chan, Samples, depthwise_max_norm=None, norm_rate=None
        )
    elif modelToEval.casefold() == 'deepconvnet':
        Mdl = zoo.DeepConvNet(
            nb_classes, Chan, Samples, kernLength = 10, F = 25, Pool = 3,
            stride = 3, batch_momentum = 0.1, dropRate = 0.5,
            max_norm = None, max_dense_norm = None
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
        loader2eval    = biasloader, 
        Model          = Mdl, 
        device         = device,
        nb_classes     = nb_classes,
        return_scores  = True,
        verbose        = verbose,
        plot_confusion = False,
        class_labels   = class_labels
    )

    scores['test_scores'] = get_performances(
        loader2eval    = testloader, 
        Model          = Mdl, 
        device         = device,
        nb_classes     = nb_classes,
        return_scores  = True,
        verbose        = verbose,
        plot_confusion = False,
        class_labels   = class_labels
    )

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
    if taskToEval.casefold() == 'alzheimer':
        start_piece_mdl = 'AlzClassification/Models/'
        start_piece_res = 'AlzClassification/Results/'
        task_piece = 'alz'
    elif taskToEval.casefold() == 'bci':
        start_piece_mdl = 'BCIClassification/Models/'
        start_piece_res = 'BCIClassification/Results/'
        task_piece = 'bci'
    elif taskToEval.casefold() == 'parkinson':
        start_piece_mdl = 'PDClassification/Models/'
        start_piece_res = 'PDClassification/Results/'
        task_piece = 'pds'
    
    if modelToEval.casefold() == 'shallownet':
        mdl_piece = 'shn'
    elif modelToEval.casefold() == 'eegnet':
        mdl_piece = 'egn'
    elif modelToEval.casefold() == 'deepconvnet':
        mdl_piece = 'dcn'
    elif modelToEval.casefold() == 'resnet':
        mdl_piece = 'res'

    if pipelineToEval.casefold() == 'raw':
        pipe_piece = 'raw'
    elif pipelineToEval.casefold() == 'filt':
        pipe_piece = 'flt'
    elif pipelineToEval.casefold() == 'ica':
        pipe_piece = 'ica'
    elif pipelineToEval.casefold() == 'icasr':
        pipe_piece = 'isr'
    
    if downsample:
        freq_piece = '125'
    else:
        if taskToEval.casefold() == 'bci':
            freq_piece = '160'
        else:
            freq_piece = '250'

    start_piece_mdl = start_piece_mdl + 'NNLNSO' + os.sep
    start_piece_res = start_piece_res + 'NNLNSO' + os.sep
    cv_piece = 'nnlns'

    rep_piece = str(repFold+1).zfill(3)
    out_piece = str(outerFold+1).zfill(3)
    in_piece = str(innerFold+1).zfill(3)
    lr_piece = str(int(lr*1e6)).zfill(6)
    chan_piece = str(Chan).zfill(3)
    win_piece = str(round(window)).zfill(3)
    
    file_name = '_'.join(
        [cv_piece, task_piece, pipe_piece, freq_piece, mdl_piece, rep_piece, 
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
    Mdl.eval()
    Mdl.to(device='cpu')
    torch.save(Mdl.state_dict(), model_path)
    
    # Save the scores
    with open(results_path, 'wb') as handle:
        pickle.dump(scores, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    if verbose:
        print('run complete')
