# How important is data partitioning in EEG deep learning applications?

This is the official repository for the research paper 

    The role of data partitioning on the performance of EEG-based deep
    learning models in supervised cross-subject analysis: a preliminary study

Published in Computers in Biology and Medicine.

[![DOI:10.1016/j.compbiomed.2025.110608](https://img.shields.io/badge/DOI-10.1016/j.compbiomed.2025.110608-blue)](https://www.sciencedirect.com/science/article/pii/S001048252500959X)

In this work, we have investigated the the role of data
partition on the performance assessment of EEG deep learning
models through cross-validation analysis.
Five distinct cross-validation strategies that operate either at the
sample or at subject level are compared across three representative
clinical and non-clinical classification tasks, using
four established DL architectures with increased complexity.
The analysis of more than 100000 different
trained models revealed strong differences between
sample-based and subject-based approaches (e.g., Leave-N-Subjects-Out),
highlighting how subject-specific characteristics can
be learned and leveraged during inference to inflate performance
estimates. Such findings confirm the necessity
of using Leave-Subject-Out strategies, particularly in clinical
applications, where subject IDs and health status are uniquely
identified.
Additionally, the analysis stressed the importance of
maintaining independent validation and test sets to respectively
monitor the training and evaluating the model. Consequently,
Nested-Leave-N-Subject-Out (N-LNSO) was found
to be sole method capable of preventing data leakage and
providing more accurate estimation of model performance
while accounting for the high inter-subject variability
inherent to EEG signals.

## How was the comparison designed

The paper describes in the detail the experimental methodology. 
Markdown files in the docs folder provide additional information
on the provided code.
Here, we report a brief description of the key points.

### Models and Tasks

We used three different tasks, covering two
clinical and one non-clinical use cases,
and three different deep learning architectures.

Tasks:
* **BCI**: motor or movement imagery classification, left and right hand.
  A famous BCI application largely studied in the domain.
* **Parkinson**, **Alzheimer**: two and three classes
  pathology classification focused on relevant neurodegenerative
  diseases.
  
Models:
* **ShallowConvNet**
* **EEGNet**
* **DeepConvNet**
* **T-ResNet**

### Model Evaluation

Data were partitioned using five distinct Cross-Validation methods.
They can be grouped in three main categories, schematized in the figure below.
Each model was evaluated using the balance accuracy
metric

<div align="center">
  <img src="Images/CV_scheme.png" width="600">
</div>

### Results

We looked for differences between the investigated CV methods by performing multiple quantitative comparisons.
Results are presented in the paper.
Here some example.

1) K-Fold vs Leave-N-Subject-Out

<div align="center">
  <img src="Images/Kfold_vs_LNSO_model_ShallowConvNet.png" width="500">
</div>

2) Leave-N-Subject-Out vs Nested-Leave-N-Subject-Out

<div align="center">
  <img src="Images/LNSO_vs_NLNSO_model_EEGNet_all_tasks.png" width="500">
</div>

3) Leave-One-Subject-Out vs Nested-Leave-One-Subject-Out

<div align="center">
  <img src="Images/LOSO_vs_FNLOSO_model_ShallowConvNet_tasks_Alzheimer.png">
</div>

## Provided code

The scripts used to generate the results presented in the paper
are available in this repository, which is derived from the following
[GitHub repo](https://github.com/MedMaxLab/eegprepro)
associated with another study we published.
Additional instructions on how to replicate
our experimental pipeline are provided in the
[docs](https://github.com/MedMaxLab/eegpartition/tree/main/docs) folder.

## Results

Performance metrics of each trained model are collected and organized in
the **ResultsTable.csv** file.
Due to the large number of training instances, model weights and results are not directly stored in this repository.
We plan to release them in zenodo.

## Authors and Citation

If you find the codes and results useful for your research,
Please consider citing our work.
It would help us continue our research.
Currently, the paper has undergone two rounds of revision.
The current version is available on Arxiv.

Contributors:

- Eng. Federico Del Pup
- M.Sc. Andrea Zanola
- M.Sc. Louis Fabrice Tshimanga
- Prof. Alessandra Bertoldo
- Prof. Livio Finos
- Prof. Manfredo Atzori

## License

The code is released under the MIT License
