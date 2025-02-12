import argparse
import subprocess
import time
from AllFnc.utilities import (
    restricted_float,
    positive_float,
    positive_int_nozero,
    positive_int,
    makeGrid,
)


def run_single_training(arg_dict):
    # create args string
    arg_str = " -d " + arg_dict["dataPath"] + \
    " -p " + arg_dict["pipelineToEval"] + \
    " -t " + arg_dict["taskToEval"] + \
    " -m " + arg_dict["modelToEval"] + \
    " -k " + arg_dict["kfoldstrat"] + \
    " -s " + str(arg_dict["downsample"]) + \
    " -z " + str(arg_dict["z_score"]) + \
    " -b " + str(arg_dict["batchsize"]) + \
    " -o " + str(arg_dict["overlap"]) + \
    " -w " + str(arg_dict["workers"]) + \
    " -v " + str(arg_dict["verbose"]) + \
    " -g " + str(arg_dict["gpudevice"]) + \
    " -l " + str(arg_dict["lr"]) + \
    " -i " + str(arg_dict["inner"]) + \
    " -f " + str(arg_dict["outer"]) + \
    " -r " + str(arg_dict["rem_interp"]) + \
    " -c " + str(arg_dict["window"])
    p = subprocess.run("python3 RunSplitTrain.py" + arg_str, shell=True, 
                       check=True, timeout = 1200)    
    return


if __name__ == '__main__':

    help_d = """
    RunKfold runs a set of trainings based on all the possible combinations
    of values written in the 'PIPE_args' dictionary.
    To keep the code base similar to other scripts of the RunKfold family,
    the path can be given as usual.
    Other parameters can be set by manually changing the code base.
    If a run fails you can restart the code and give the starting index of the
    for loop.
    This is for the Sample-based K-Fold CV.
    
    Example of first call:
    
    $ Python RunKfold -d /path/to/data

    Example of another call if run fails for some reasons:

    $ Python RunKfold -d /path/to/data -s 130
    
    """
    parser = argparse.ArgumentParser(description=help_d)
    parser.add_argument(
        "-d",
        "--datapath",
        dest      = "dataPath",
        metavar   = "datasets path",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = None,
        help      = """
        The dataset path. This is expected to be static across all trainings. 
        dataPath must point to a directory which contains four subdirecotries, 
        one with all the pickle files containing EEGs preprocessed with a 
        specific pipeline. Subdirectoties are expected to have the following names, 
        which are the same as the preprocessing pipelinea to evaluate:
        1) raw; 2) filt; 3) ica; 4) icasr
        """,
    )
    parser.add_argument(
    "-s",
    "--start",
    dest      = "start_idx",
    metavar   = "starting index",
    type      = positive_int,
    nargs     = '?',
    required  = False,
    default   = 0,
    help      = """
    The starting index. It can be used to restart the training if one failed
    or stopped for some reasons. 
    """
)

    # basically we overwrite the dataPath if something was given
    args = vars(parser.parse_args())
    dataPathInput = args['dataPath']
    StartIdx = args['start_idx']
    if dataPathInput is None:
        dataPathInput = '/data/delpup/datasets/eegpickle/'
    
    pipes = ["filt",         "ica",       "icasr"]
    tasks = ["bci",    "parkinson",   "alzheimer"]

    arg_list = []
    for pipe, task in zip(pipes,tasks):
        PIPE_args = {
            "dataPath": [dataPathInput],
            "pipelineToEval": [pipe],
            "taskToEval": [task],
            "modelToEval": ["shallownet", "eegnet", "deepconvnet", "resnet"],
            "kfoldstrat": ["kfold"],
            "downsample": [True],
            "z_score": [True],
            "rem_interp": [True],
            "batchsize": [64],
            "window": [4.0],
            "overlap": [0.0],
            "workers": [0],
            "gpudevice": ["cuda:0"],
            "verbose": [False],
            "lr": [0.0],
            "inner": [1],
            "outer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        # create the argument grid and discard impossible combinations
        arg_list += makeGrid(PIPE_args)

    # print the final dictionary
    print("running trainings with the following set of parameters:")
    print("note: combinations with subject False and appleloss True will be discarded")
    print(" ")
    for key in PIPE_args:
        if key == 'pipelineToEval':
            print( f"{key:15} ==> {pipes}")
        elif key == 'taskToEval':
            print( f"{key:15} ==> {tasks}") 
        elif key == 'inner':
            print( f"{key:15} ==> no inner folds")
        else:
            print( f"{key:15} ==> {PIPE_args[key]}") 
    
    # Run each training in a sequential manner
    N = len(arg_list)
    print(f"the following setting requires to run {N:5} trainings")
    if StartIdx>0:
        print(f"Restart from training number {StartIdx:5}")
        StartIdx = StartIdx - 1
    
    for i in range(StartIdx, N):
        print(f"running training number {i+1:<5} out of {N:5}")
        Tstart = time.time()
        run_single_training(arg_list[i])
        Tend = time.time()
        Total = int(Tend - Tstart)
        print(f"training performed in    {Total:<5} seconds")
    
    print(f"Completed all {N:5} trainings")
    # Just a reminder to keep your GPU cool
    if (N-StartIdx)>1000:
        print(f"...Is your GPU still alive?")
