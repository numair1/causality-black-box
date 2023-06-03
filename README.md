# causality-black-box

This is the official repository for the paper titled "Explaining The Behavior Of Black Box Algorithms Using Causal Learning".

This repository has the following three folders, each containing code on different aspects of paper, listed below:

1. /simulation - code for the HydraNet implmentation, along with the subsequent logistic regression and causal discovery parts of the simulation.
2. /birds - code for the ResNet18 pretrained classification model, output of causal discovery datasets, as well as the implementation of lime and SHAP.
3. /XR - code for XR example, along with the relevant LIME and SHAP implementations.  

## Simulation

This folder contists of two groups of files, the first group helping train the HydraNet network, and the second group helping combine it with the logistic regression to create a black box model with established ground truths. Each of these groups is described below

### HydraNet Files

This group of files serves to train a HydraNet model, and save the output as HydraNet.pth. The following files are used in this step:

- HydraNet_data: Directory containing two sub-directories, train_data and val_data, and each of these contains images used to train HydraNet. The images contain the following shapes - horizontal bar, vertical bar, circle and triangle, and each of these is generated using an independent unbiased coin flip. 
- train_net.ipynb: This is a jupyter notebook that uses the above training data to train HydraNet and save the trained model as HydraNet.pth.
- simulate_HydraNet.py: File that generates the training and validation datasets to train the HydraNet model.
- HydraNet.pth - The saved torch model resulting from the training procedure.

### Full Black Box Model Files

This is the group of files that implement the HydraNet + logistic regression to create a full black-box model, and then 
generate predictions on images that are subsequently used in TETRAD with the FCI algorithm to learn a causal graph.
- data_full_pipeline: This directory contains 20k images, each containing horizontal bars, vertical bars, circles and 
triangles simulated according to the DGP outlined in simulate_from_DGP.py.
- simulate_from_DGP.py : generates images following the DGP outlined in documentation/simulation.pdf.
- HydraNet_logistic_regression.ipynb: File that loads the saved HydraNet model, and then generates predictions on the 
presence of horizontal bars, vertical bars, triangles and circles in each image in data_full_pipeline. The results of 
these predictions is saved in HydraNet_predictions_data_full_pipeline, and the _hn suffix means that the feature annotation
is generated from a HydraNet prediction. Next, using the true y and the shape predictions from HydraNet, a logistic regression
model is trained, and then used to generate predictions y_hat. This file is saved as black_box_predictions_data_full_pipeline.csv
- black_box_predictions_data_full_pipeline.csv: This file is then used as input to TETRAD to run the FCI algorithm at 
0.005 significance, and with horizontal bar excluded in order to obtain the final graph.

#### Hyperparameter Settings

TO DO: Add in details regarding the setting of various hyperparameters for replicability. 