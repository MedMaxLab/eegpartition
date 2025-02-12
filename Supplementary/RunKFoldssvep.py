import sys
sys.path.append('..')
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
    " -m " + arg_dict["modelToEval"] + \
    " -k " + arg_dict["kfoldstrat"] + \
    " -z " + str(arg_dict["z_score"]) + \
    " -b " + str(arg_dict["batchsize"]) + \
    " -v " + str(arg_dict["verbose"]) + \
    " -g " + str(arg_dict["gpudevice"]) + \
    " -l " + str(arg_dict["lr"]) + \
    " -i " + str(arg_dict["inner"]) + \
    " -f " + str(arg_dict["outer"])
    p = subprocess.run("python3 RunSplitTrainssvep.py" + arg_str, shell=True, 
                       check=True, timeout = 3000)    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datapath",
        dest      = "dataPath",
        metavar   = "datasets path",
        type      = str,
        nargs     = '?',
        required  = False,
        default   = None,
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
)

    # basically we overwrite the dataPath if something was given
    args = vars(parser.parse_args())
    dataPathInput = args['dataPath']
    StartIdx = args['start_idx']
    if dataPathInput is None:
        dataPathInput = '/data/delpup/datasets/eegpickle/'
    
    PIPE_args = {
        "dataPath": [dataPathInput],
        "modelToEval": ["shallownet", "eegnet", "deepconvnet", "resnet"],
        "kfoldstrat": ["kfold"],
        "z_score": [True],
        "batchsize": [64],
        "gpudevice": ["cuda:1"],
        "verbose": [False],
        "lr": [0.0],
        "inner": [1],
        "outer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    # create the argument grid and discard impossible combinations
    arg_list = makeGrid(PIPE_args)

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
