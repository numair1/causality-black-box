# Explaining the Behavior of Black-Box Prediction Algorithms with Causal Learning

# Table Of Contents

1. [Introduction](#introduction)
2. [Simulation](#simulation)
3. [Birds Data Example](#birds-data-example)
4. [XR Data Example](#xr-data-example)
5. [Documentation](#documentation)

## Introduction

This is the official repository for the code used in the simulation and data applications for the paper "Explaining The Behavior Of Black Box Algorithms Using Causal Learning" by Sani, Malinksy and Shpitser (2025). The paper can be found [here](https://arxiv.org/pdf/2006.02482).

This repository contains the code for the simulation in Section 5, as well as the code for the experiments in Section 6 of the paper. The documentation directory contains miscellaneous notes and calculations for the simulation and examples used in the paper.  

## Simulation

This folder contists of the code for the simulation in Section 5 of the paper. It contains two groups of files, the first group of files contain code for helping train the HydraNet network, the original reference and code for which can be found [here](https://pyimagesearch.com/2022/08/17/multi-task-learning-and-hydranets-with-pytorch/). The second group of files helps combine it with the logistic regression to create a black box model that behaves in the desired way as described in the simulation details. Each of these groups of files is described below.

### HydraNet Files

This group of files serves to train a HydraNet model, and save the output as HydraNet.pth. The following files are used in this step:

- HydraNet_data: Directory containing two sub-directories, train_data and val_data, and each of these contains images used to train HydraNet. The images contain the following shapes - horizontal bar, vertical bar, circle and triangle, and each of these is generated using an independent unbiased coin flip. 
- train_net.ipynb: This is a jupyter notebook that uses the above training data to train HydraNet and save the trained model as HydraNet.pth.
- simulate_HydraNet.py: File that generates the training and validation datasets to train the HydraNet model.
- HydraNet.pth - The saved torch model resulting from the training procedure.

### Full Black Box Model Files

This is the group of files that implement the HydraNet + logistic regression combination to create a full black-box prediction model that behaves in a way as described in the simulation, and then generates predictions on images that are subsequently used in TETRAD with the FCI algorithm to learn the causal graph displayed in Figure 7(b). The follwiing files and directories are used in this step:

- simulate_from_DGP.py : Generates images following the DGP outlined in the following [documentation/simulation.pdf](documentation/simulation.pdf) and stores it in [simulation/data_full_pipeline](simulation/data_full_pipeline).

- data_full_pipeline: This directory contains 20k images, each containing horizontal bars, vertical bars, circles and 
triangles simulated according to the DGP outlined in simulate_from_DGP.py.

- HydraNet_logistic_regression.ipynb: File that loads the saved HydraNet model, and then generates predictions on the 
presence of horizontal bars, vertical bars, triangles and circles in each image in data_full_pipeline. The results of 
these predictions is saved in [HydraNet_predictions_data_full_pipeline.csv](simulation/HydraNet_predictions_data_full_pipeline.csv), and the _hn suffix means that the feature annotation
is generated from a HydraNet prediction. Next, using the true y and the shape predictions from HydraNet, a logistic regression
model is trained, and then used to generate predictions y_hat. This file is saved as [black_box_predictions_data_full_pipeline.csv](simulation/black_box_predictions_data_full_pipeline.csv).


- black_box_predictions_data_full_pipeline.csv: This file is then used as input to [TETRAD GUI](https://www.cmu.edu/dietrich/philosophy/tetrad/use-tetrad/tetrad-application.html) to run the FCI algorithm at 
0.005 significance, with the v_bar,circle,triangle along with y_hat being used, as the nodes and with horizontal bar excluded in order to obtain the final graph. Background knowledge that y_hat is a descendant of the features is also utilized. 

## Birds Data Example

The dataset for this example can be downloaded at https://www.vision.caltech.edu/datasets/cub_200_2011/. 

### Data Cleaning
Once the daatast is downloaded, it can be cleaned using the code below.   

- data_cleaning/group_classes.py: Takes the CUB dataset where folders containing images of different species of birds have
been grouped into a bigger folder representing the coarse grouping, and the code copies all of the images in these species 
sub directories into the coarse grouping directory. The 9 categories we pick are: 
1. Flycatcher
2. Gull
3. Kingfisher
4. Sparrow 
5. Tern 
6. Vireo 
7. Warbler 
8. Woodpecker 
9. Wren

The output of this is saved in data/consolidated_dataset. This gives an image dataset containing 5514 images.
- data_cleaning/balance_classes.py: Downsamples the over-represented classes outputted from the above coarse grouping saved the directory
data/consolidated_dataset. Sparrow and Warbler are overrepresented and consequently downsampled. 
- data_cleaning/split_data.py: Splits the dataset into a 70-15-15 train/val/test split.
- data_cleaning/colorize.py: Discards .DS_Store files as well as discrads images that are in grayscale.

### Model Training

The model is trained on the ResNet18 architecture with pre-trained weights loaded, and then fine tuned on the data. The [train_net.py](birds/network_training/train_net.py) file contains the relevant code for training the model, as well as saving the predictions for each imagine in [model_preds.txt](birds/network_training/model_preds.txt).

### Structure Learning

The following files are involved in the final pipeline:

- causal_structure_learning/create_TETRAD_dataset.py: Takes the image attributes in data/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt, coarsens them accroding to the grouping outlined in Appendix C.2, and then joins with the model_preds.txt to output [birds_dataset.txt](birds/causal_structure_learning/birds_dataset.txt).
  
- causal_structure_learning/subsample_data.py: Performs sampling with replacement to generated bootstrapped datasets based on birds_dataset.txt. These bootstrapped datasets can be found in the birds/causal_structure_learning/subsampled_datasets directory.
  
- FCI.sh and run_FCI.py: Bash script and Python script respectively that apply the FCI algorithm to all the datasets in the subsampled_datasets directory and saves the output of the FCI algorithm in [FCI_birds_05_depth4](birds/causal_structure_learning/FCI_birds_05_depth4), with an individual FCI file corresponding to each dataset.

- compute_importance.py: Parses the output of the FCI algorithm and computes the importance of each features and displays it as a dictionary.

- plot_bar_graphs.py: Takes the resulting importances computed for both the birds and the X-Ray example creates bar graphs used in Figure 8.

## XR Data Example

### Data Cleaning

### Model Training

### Structure Learning


## Documentation

This directory contains calculations for the SHAP counterexample in Example 1, as well as the data generation process used in the simulation.

