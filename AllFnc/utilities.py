import argparse
import glob
import itertools
import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats
import glob

__all__ = [
    "makeGrid",
    "str2bool",
    "restricted_float",
    "positive_float",
    "positive_int_nonzero",
    "positive_int",
    "column_switch",
    "gather_results",
    "GetLrDict",
    "GetLearningRateString",
    "get_full_name",
    "qcd",
    "ztest",
    "signtest",
    "gather_metric_values",
    "convert_performance_totable"
]


def makeGrid(pars_dict):
        keys = pars_dict.keys()
        combinations = itertools.product(*pars_dict.values())
        ds = [dict(zip(keys, cc)) for cc in combinations]
        return ds


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.casefold() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.casefold() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0)" % (x,))
    return x


def positive_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x


def positive_int_nozero(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an integer" % (x,))

    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x


def positive_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an integer" % (x,))

    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive value" % (x,))
    return x


def column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df.reindex(columns = i)
    return df

def gather_results(save = False, filename = None):
    metrics_list = [ 
        'accuracy_unbalanced', 'accuracy_weighted',      'precision_micro',
        'precision_macro',    'precision_weighted',         'recall_micro',
        'recall_macro',          'recall_weighted',        'f1score_micro',
        'f1score_macro',        'f1score_weighted',         'rocauc_micro',
        'rocauc_macro',          'rocauc_weighted',          'cohen_kappa'
    ]
    piece_list = [
        'validation_type',          'task',             'pipeline',
        'sampling_rate',           'model',      'repetition_fold',
        'outer_fold',         'inner_fold',        'learning_rate',
        'channels',               'window',
    ]
    file_list = glob.glob('**/Results/**/*.pickle')
    results_list = [None]*len(file_list)
    for i, path in enumerate(file_list):

        # Get File name
        file_name = path.split(os.sep)[-1]
        file_name = file_name[:-7]

        # Get all name pieces
        pieces = file_name.split('_')

        if pieces[0] != 'nnlns':
            # convert to numerical some values
            for k in [3,5,6,7,8,9]:
                pieces[k] = int(pieces[k])
                if k == 7:
                    pieces[k] = pieces[k]/1e6
            pieces.insert(5, 1)
        else:
            for k in [3,5,6,7,8,9,10]:
                pieces[k] = int(pieces[k])
                if k == 8:
                    pieces[k] = pieces[k]/1e6

        # open results
        with open(path, "rb") as f:
            mdl_res = pickle.load(f)

        # append results
        for metric in metrics_list:
            pieces.append(mdl_res[metric])

        # final list
        results_list[i] = pieces

    # convert to DataFrame and swap two columns for convenience
    results_table = pd.DataFrame(results_list, columns= piece_list + metrics_list)
    results_table = column_switch( results_table, 'model', 'sampling_rate')
    results_table.sort_values(
        ['validation_type', 'model', 'task', 'pipeline', 'repetition_fold', 'outer_fold', 'inner_fold'],
        ascending=[True, True, True, True, True, True, True],
        inplace=True
    )
    results_table.reset_index(drop=True, inplace=True)

    # store if required
    if save:
        if filename is not None:
            if filename[:-3] == 'csv':
                results_table.to_csv(filename, index=False)
            else:
                results_table.to_csv(filename + '.csv', index=False)
        results_table.to_csv('ResultsTable.csv', index=False)
    return results_table


def GetLrDict():
    lr_dict = {
        'shallownet': {
            'eyes': 1e-03,
            'parkinson': 2.5e-04, #2.5e-05
            'alzheimer': 5e-05,
            'bci': 7.5e-04
        },
        'deepconvnet': {
            'eyes': 7.5e-04,
            'parkinson': 2.5e-04,
            'alzheimer': 7.5e-04,
            'bci': 7.5e-04
        },
        'resnet': {
            'eyes': 5e-04,
            'parkinson': 1e-05,
            'alzheimer': 5e-05,
            'bci': 5e-4
        }
    }
    return lr_dict

    
def GetLearningRateString(model, task):
    model_conversion_dict = {
        'shn': 'shallownet',
        'dcn': 'deepconvnet',
        'res': 'resnet'
    }
    task_conversion_dict = {
        'alz': 'alzheimer',
        'bci':'bci',
        'pds': 'parkinson'
    }
    if len(model)==3:
        model = model_conversion_dict.get(model)
    if len(task)==3:
        task = task_conversion_dict.get(task)
    lr_dict = GetLrDict()
    lr = lr_dict.get(model).get(task)
    lr = str(int(lr*1e6)).zfill(6)
    return lr


def get_full_name(acronym):
    conversion_dict = {
        'shn': 'ShallowConvNet',
        'dcn': 'DeepConvNet',
        'res': 'T-ResNet',
        'alz': 'Alzheimer', 
        'bci': 'BCI',
        'pds': 'Parkinson',
        'KFOLD': 'Sample-based Cross-Validation',
        'LNSO': 'Leave-N-Subjects-Out',
        'LOSO': 'Leave-One-Subject-Out',
        'NLNSO': 'Nested-Leave-N-Subjects-Out',
        'NLOSO': 'Nested-Leave-One-Subject-Out',
        'NKFOLD': 'Nested Sample-Based Cross-Validation',
        'PNLOSO': 'Partial Nested-Leave-One-Subject-Out',
        'NNLNSO': 'Nested-Nested-Leave-N-Subject-Out'
    }
    return conversion_dict.get(acronym)


def gather_metric_values(metric, task, model, partition, validation=False, test=False):
    
    performances = {i: {j: {k:[] for k in partition} for j in model} for i in task}
    if validation:
        performances_val = {i: {j: {k:[] for k in partition} for j in model} for i in task}
    if test:
        performances_tst = {i: {j: {k:[] for k in partition} for j in model} for i in task}
    
    for i in task:
        if i =='pds': 
            ch = '032'
        elif 'al' in i:
            ch = '019'
        else:
            ch = '061' 
        for j in model:
            for k in partition:
                file_name = '_'.join(['*', i, '*', j,  '*', GetLearningRateString(j,i), ch, '*'])
                file_names = glob.glob('**/Results/' + k + os.sep + file_name + '.pickle')
                if k == 'NNLNSO':
                    file_names = sorted(
                        file_names,
                        key = lambda x: (
                            int(x.split(os.sep)[-1].split('_')[5]),
                            int(x.split(os.sep)[-1].split('_')[6]),
                            int(x.split(os.sep)[-1].split('_')[7])
                        )
                    )
                else:
                    file_names = sorted(
                        file_names,
                        key = lambda x: (
                            int(x.split(os.sep)[-1].split('_')[5]),
                            int(x.split(os.sep)[-1].split('_')[6])
                        )
                    )
                if len(file_names) > 0:
                    performances[i][j][k] = [None]*len(file_names)
                    if validation:
                        performances_val[i][j][k] = [None]*len(file_names)
                    if test:
                        performances_tst[i][j][k] = [None]*len(file_names)
                    for n, f in enumerate(file_names):
                        with open(f, 'rb') as file:
                            ith_result=pickle.load(file)
                        performances[i][j][k][n] = ith_result[metric]*100
                        if validation:
                            try:
                                performances_val[i][j][k][n] = ith_result['validation_scores'][metric]*100
                            except Exception:
                                performances_val[i][j][k][n] = performances[i][j][k][n]
                        if test:
                            try:
                                performances_tst[i][j][k][n] = ith_result['test_scores'][metric]*100
                            except Exception:
                                performances_tst[i][j][k][n] = performances[i][j][k][n]
    if validation:
        if test:
            return performances, performances_val, performances_tst
        else:
            return performances, performances_val
    else:
        return performances


def convert_performance_totable(
    performance_dict,
    performance_dict_val=None,
    performance_dict_tst=None
):
    
    Nsubj = {
        'alz': 88,
        'pds': 81,
        'bci': 106
    }
    df = 0
    cdf = 0
    # i == task, j == model, k = partition
    for i in performance_dict:
        for j in performance_dict[i]:
            for k in performance_dict[i][j]:
                df = None
                if k in ['KFOLD', 'LNSO']:
                    if len(performance_dict[i][j][k]) == 10:
                        df = pd.DataFrame(
                            performance_dict[i][j][k],
                            columns=['Metric']
                        ).assign(Task=i).assign(Model=j).assign(Partition=k)
                        df = df.assign(Inner = -1)
                        df = df.assign(Outer=[n for n in range(1,11)])
                        df = df.assign(Repetition=[1 for n in range(len(performance_dict[i][j][k]))])
                elif k == 'LOSO':
                    if len(performance_dict[i][j][k]) == Nsubj[i]:
                        df = pd.DataFrame(
                            performance_dict[i][j][k],
                            columns=['Metric']
                        ).assign(Task=i).assign(Model=j).assign(Partition=k)
                        df = df.assign(Inner = -1)
                        df = df.assign(Outer=[n for n in range(1, Nsubj[i]+1)])
                        df = df.assign(Repetition=[1 for n in range(len(performance_dict[i][j][k]))])
                elif k == 'NKFOLD':
                    if len(performance_dict[i][j][k]) == 100:
                        df = pd.DataFrame(
                            performance_dict[i][j][k],
                            columns=['Metric']
                        ).assign(Task=i).assign(Model=j).assign(Partition=k)
                        df = df.assign(Inner = [n for n in range(1, 11)]*10)
                        df = df.assign(Outer=[n//10+1 for n in range(100)])
                        df = df.assign(Repetition=[1 for n in range(len(performance_dict[i][j][k]))])
                elif k == 'NLNSO':
                    if len(performance_dict[i][j][k]) == 100:
                        df = pd.DataFrame(
                            performance_dict[i][j][k],
                            columns=['Metric']
                        ).assign(Task=i).assign(Model=j).assign(Partition=k)
                        df = df.assign(Inner = [n for n in range(1, 11)]*10)
                        df = df.assign(Outer=[n//10+1 for n in range(100)])
                        df = df.assign(Repetition=[1 for n in range(len(performance_dict[i][j][k]))])
                elif k == 'NLOSO':
                    if len(performance_dict[i][j][k]) == (Nsubj[i]*(Nsubj[i]-1)):
                        df = pd.DataFrame(
                            performance_dict[i][j][k],
                            columns=['Metric']
                        ).assign(Task=i).assign(Model=j).assign(Partition=k)
                        df = df.assign(Inner = [n for n in range(1, Nsubj[i])]*Nsubj[i])
                        df = df.assign(Outer = [n//(Nsubj[i]-1)+1 for n in range(Nsubj[i]*(Nsubj[i]-1))])
                        df = df.assign(Repetition=[1 for n in range(len(performance_dict[i][j][k]))])
                elif k == 'PNLOSO':
                    if len(performance_dict[i][j][k]) == (Nsubj[i]*10):
                        df = pd.DataFrame(
                            performance_dict[i][j][k],
                            columns=['Metric']
                        ).assign(Task=i).assign(Model=j).assign(Partition=k)
                        df = df.assign(Inner = [n for n in range(1, 11)]*Nsubj[i])
                        df = df.assign(Outer = [n//10+1 for n in range(Nsubj[i]*10)])
                        df = df.assign(Repetition=[1 for n in range(len(performance_dict[i][j][k]))])
                elif k == 'NNLNSO':
                    if len(performance_dict[i][j][k]) == 1000:
                        df = pd.DataFrame(
                            performance_dict[i][j][k],
                            columns=['Metric']
                        ).assign(Task=i).assign(Model=j).assign(Partition=k)
                        df = df.assign(Inner = [n for n in range(1, 11)]*100)
                        df = df.assign(Outer=[n//10+1 for n in range(100)]*10)
                        df = df.assign(Repetition=[n//100+1 for n in range(1000)])
                if not(isinstance(cdf, pd.DataFrame)):
                    cdf = df
                else:
                    try:
                        cdf = pd.concat([cdf, df])
                    except:
                        cdf = df
    cdf = cdf.dropna(axis=1)
    if performance_dict_val is not None:
        cdf_val = convert_performance_totable(performance_dict_val)
        cdf["MetricVal"] = cdf_val["Metric"]
    if performance_dict_tst is not None:
        cdf_tst = convert_performance_totable(performance_dict_tst)
        cdf["MetricTest"] = cdf_tst["Metric"]
    return cdf


def qcd(x, axis=0):
    num = ( np.percentile(x, 75, axis) - np.percentile(x, 25, axis) )
    den = ( np.percentile(x, 75, axis) + np.percentile(x, 25, axis) )
    return num/den


def ztest(b1, se1, b2=1, se2=0):
    return 2*stats.norm.cdf(-np.abs((b1 - b2)/np.sqrt( se1**2 + se2**2 )))


def signtest(x1, x2, alpha=0.05):
    diff = x1-x2
    n_pos = (diff>0).sum()
    n_neg = (diff<0).sum()
    n = n_pos + n_neg
    x = min(n_pos, n_neg)
    pvalue = 2*stats.binom.cdf(x, n, 0.5)
    print(f"sign test: alpha is {alpha:.4f}, non-zero differences are {n:.3f}, ")
    print(f"           statistic is {x:.3f}, pvalue is {pvalue:.5f}")
    if pvalue > alpha:
        print("           pvalue is larger than alpha. The testing is not significant.")
    else:
        print("           pvalue is smaller than alpha.")
        print("           The control experiment is significantly better than test.")
    return pvalue