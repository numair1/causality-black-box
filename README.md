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

### Full Black Box Model Files

- data_full: This directory contains 10k images, each containing horizontal bars, vertical bars, circles and triangles simulated from an unbiased coin toss.
- train_net.ipyn: Jupyter notebook tha trains the HydraNet model and saves it as HydraNet.pth.
- train_data/val_data: Data generated according to the data generating process described in the paper, TO DO: identify the correct file associated with this simulation, and re-name things accordingly.
