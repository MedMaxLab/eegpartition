# How important is data partition in EEG deep learning applications?

This is the official repository for the research paper 

    Rethinking how to evaluate model performance
    in deep learning-based electroencephalogram analysis 

Submitted to Biomedical Signal Processing and Control.

In this work, we have investigated the the role of data
partition on the performance assessment of EEG deep learning
models through cross-validation analysis.
Five distinct cross-validation strategies that operate either at the
sample or at subject level are compared across three representative
clinical and non-clinical classification tasks, using
three established DL architectures with increased complexity.
The analysis of almost one hundred thousand different
trained models revealed strong differences between
sample-based and subject-based approaches (e.g., Leave-N-Subject-Out),
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
* **DeepConvNet**
* **T-ResNet**

### Model Evaluation

Data were partitioned using five distinct Cross-Validation methods.
They can be grouped in three main categories, schematized in the figure below.
Each model was evaluated using the balance accuracy
metric

<div align="center">
  <img src="Images/CV_scheme.pdf" width="600">
</div>

### Results

We looked for differences between the investigated CV methods by performing multiple quantitative comparisons.
Results are presented in the paper.
Here some example.

1) K-Fold vs Leave-N-Subject-Out

<div align="center">
  <img src="Images/LightTheme/Kfold_vs_LNSO_model_ShallowConvNet.pdf" width="500">
</div>

2) Leave-N-Subject-Out vs Nested-Leave-N-Subject-Out

<div align="center">
  <img src="Images/LightTheme/LNSO_vs_NLNSO_model_T-ResNet_all_tasks.pdf" width="500">
</div>

3) Leave-One-Subject-Out vs Nested-Leave-One-Subject-Out

<div align="center">
  <img src="Images/LightTheme/LOSO_vs_FNLOSO_model_T-ResNet_tasks_Parkinson.pdf">
</div>

## Provided code

Scripts used to generate the results presented in the paper
are available in this repository.
Additional instructions on how to replicate
our experimental pipeline are provided in the
[docs](https://github.com/MedMaxLab/eegpartition/tree/main/docs) folder.

## Results

Performance metrics of each trained model are collected and organized in
the **ResultsTable.csv** file.
Due to the large number of training instances, model weights and results are not directly stored in this repository.
We plan to release them with zenodo, together with the paper and the code-base

## Authors and Citation

If you find codes and results useful for your research,
please concider citing our work. It would help us to continue our research.
At the moment, we are working on a research paper to submit to
Biomedical Signal Processing and Control.


Contributors:

- Eng. Federico Del Pup
- M.Sc. Andrea Zanola
- M.Sc. Louis Fabrice Tshimanga
- Prof. Alessandra Bertoldo
- Prof. Livio Finos
- Prof. Manfredo Atzori

## License

The code is released under the MIT License
