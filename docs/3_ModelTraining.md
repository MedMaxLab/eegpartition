# Model Training

Training all the models should be pretty simple but will require lot of time.
To simplify thing, all the results will be uploaded in a zenodo repository.
Here, we provide a summary table created with the ``CreateResultsTable.py`` file.
However, if you want to rerun all the training here is a description of the
scripts you need to run.

The entire process relies on the Python "Run..." files:
the ``RunSplitTrain.py``/``RunSplitTrainNN.py``
(or its ``RunKfoldComboB0X.py`` variants for multiple GPUs) 
files.

Basically, ``RunSplitTrain.py`` and ``RunSplitTrainNN.py`` runs a single training with a specific
set of arguments. Here is the script help:

```
RunSplitTrain run a single training with a specific split
extracted from a chosen cross-validation modality.
Many parameters can be set to control the training process.
All of them will be recorded using a unique file name.
The required input is the root dataset path.
Others have a default in case you want to check a single
demo run. Example:

$ python RunSplitTraining -d /path/to/data

options:
  -h, --help
        show this help message and exit
  -d datasets path, --datapath datasets path
        The dataset path. This is expected to be static across all trainings.
        dataPath must point to a directory which contains four subdirecotries,
        one with all the pickle files containing EEGs preprocessed with a
        specific pipeline. Subdirectoties are expected to have the following names,
        which are the same as the preprocessing pipelinea to evaluate:
            1) raw; 2) filt; 3) ica; 4) icasr
  -p [preprocessing pipeline], --pipeline [preprocessing pipeline]
        The pipeline to consider. It can be one of the following:
            1) raw; 2) filt; 3) ica; 4) icasr
  -t [task], --task [task]
        The task to evaluate. It can be one of the following:
            1) alzheimer; 2) parkinson; 3) bci;
  -m [model], --model [model]
        The model to evaluate. It can be one of the following:
            1) shallownet; 2) deepconvnet; 3) resnet;
  -k [kfold], --kfoldstrat [kfold]
        The model to evaluate. It can be one of the following:
            1) kfold (k-fold intra-subject)
            2) lnso (leave-n-subject-out)
            3) loso (leave-one-subject-out)
            4) ploso (partial nested-leave-one-subject-out)
            5) nkfld (nested k-fold intra subject)
            6) nlnso (nested-leave-n-subject-out)
            7) nloso (nested-leave-one-subject-out)
  -f [outer fold], --outer [outer fold]
        The outer fold to evaluate. It will be used for every possible CV strategy.
        For the N-NLSO Cross validation technique, an inner fold must also be given.
  -i [inner fold], --inner [inner fold]
        The inner fold to evaluate.
        It will be used only for the N-NLSO Cross validation technique.
  -s [downsample], --downsample [downsample]
        A boolean that set if downsampling at 125 Hz should be applied or not.
        If set to False, data at 250 Hz (160 Hz for the MI task) will be used.
  -z [zscore], --zscore [zscore]
        A boolean that set if the z-score should be applied or not.
        The presented analysis applied the z-score.
  -r [remove interpolated], --rminterp [remove interpolated]
        A boolean that set if the interpolated channels should be removed or not.
        BIDSAlign aligns all EEGs to a common 61 channel template based on the 10_10
        International System.
  -b [batch size], --batch [batch size]
        Define the Batch size. It is suggested to use 64 or 128.
        The experimental analysis was performed on batch 64.
  -o [windows overlap], --overlap [windows overlap]
        The overlap between time windows. Higher values means more samples but
        higher correlation between them. 0.25 is a good trade-off.
        Must be a value in [0,1). Here, no overlap is used.
  -l [learning rate], --learningrate [learning rate]
        The learning rate. If left to its default (zero) a proper learning rate
        will be chosen depending on the model and task to evaluate.
        Optimal learning rates were identified by running multiple trainings with
        different set of values.
        Must be a positive value
  -c [window], --cwindow [window]
        The window (input) size, in seconds. Each EEG will be partitioned in windows
        of length equals to the one specified by this input. c was the first available
        letter.
  -w [dataloader workers], --workers [dataloader workers]
        The number of workers to set for the dataloader. Datasets are preloaded
        for faster computation, so 0 is preferred due to known issues on values
        greater than 1 for some os, and to not increase too much the memory usage.
  -g [torch device], --gpu [torch device]
        The cuda device to use.
  -v [VERBOSE], --verbose [VERBOSE]
        Set the verbosity level of the whole script. If True, information about
        the choosen split, and the training progression will be displayed
```

``Run_CVMethod_.py`` creates the grid of values to parse
to ``RunSplitTrain.py`` and run all the training sequentially.
If you have multiple GPUs, you can duplicate this code
and divide the grid of possible arguments (for example,
split the outer fold numbers) to parallelize the
training instances.

Both the results on the test set and the model weights will
be stored in the relative folders. Each task has its own
folder with name ``acronym+Classification`` (e.g.,
EoecClassification).
Note that to you require more than 150 GB.