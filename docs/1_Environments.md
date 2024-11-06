# Environments

To run the experiments, you should:

1. install matlab with additional packages to preprocess EEG data
2. Create a Python environment to run
   all the trainings and to generate all the figures.

## Matlab for data preprocessing

Data were preprocessed with [BIDSAlign](https://github.com/MedMaxLab/BIDSAlign), an EEGLab extension
designed by our team for the preprocessing and alignment
of multiple EEG repositories in BIDS format
(like those in openneuro).

To install BIDSAlign, you should:

1. Install MATLAB. We have used 2023b on a Apple Silicon Mac,
   but older versions should work. Just check that EEGLAB can be
   run.
2. Install EEGLAB and add its path to the MATLAB search path.
   (Or simply run ``eeglab`` in the command window.)
   We have used EEGLAB 2024.0, but older versions should work.
4. Install all the BIDSAlign dependencies. They can be installed
   directly with EEGLAB.
   * "Biosig" v3.8.3
   * "FastICA" v25
   * "Fileio" v20240111
   * "ICLabel" v1.4
   * "MARA" v1.2
   * "Viewprops" v1.5.4
   * "bva-io" v1.73
   * "clean_rawdata" v2.91
   * "dipfit" v5.3
   * "firfilt" v2.7.1
   * "eegstats" v1.2
5. Download BIDSAlign from the following
   [github page](https://github.com/MedMaxLab/BIDSAlign)
   and add its path to the MATLAB search path.

That's it, if everything is done correctly, running
``bidsalign nogui`` on the command window will open BIDSAlign.

## Python Environment

The deep learning part mainly relies on
[selfeeg](https://github.com/MedMaxLab/selfEEG),
a Python library developed by our team and desigend to run
EEG-based deep learning experiments built on top of Pytorch.
As described in the library main page, it is suggested to first
install Pytorch using the correct command, which can vary 
depending on your OS and CUDA version, then install selfeeg.

We STRONGLY suggest to install selfeeg directly from github
with the following command:

    !pip install git+https://github.com/MedMaxLab/selfEEG

The environment used by our team specifically has:

* python = 3.11.9 (hab00c5b_0_cpython)
* pytorch = 2.4.0 (py3.11_cuda11.8_cudnn8.7.0_0)
* torchaudio = 2.4.0 (py311_cu118)
* torchvision = 0.19.0 (py311_cu118)
* scipy = 1.13.1 (py311h08b1b3b_1)
* numpy = 1.26.4 (py311h08b1b3b_0)
* pandas = 2.2.2 (py311ha02d727_0)
* jupyterlab = 4.2.4 (pyhd8ed1ab_0)
* selfeeg = 0.2.0 (from git)
