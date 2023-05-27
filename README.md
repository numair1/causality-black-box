# causality-black-box

This is the official repository for the paper titled "Explaining The Behavior Of Black Box Algorithms Using Causal Learning".

This repository has the following three folders, each containing code on different aspects of paper, listed below:

1. /simulation - code for the HydraNet implmentation, along with the subsequent logistic regression and causal discovery parts of the simulation.
2. /birds - code for the ResNet18 pretrained classification model, output of causal discovery datasets, as well as the implementation of lime and SHAP.
3. /XR - code for XR example, along with the relevant LIME and SHAP implementations.  

## Simulation

This folder contains each of the files and directiories described below

- data_full: This directory contains 10k images, each containing horizontal bars, vertical bars, circles and triangles simulated from an unbiased coin toss.
- train_net.ipyn: Jupyter notebook tha trains the HydraNet model and saves it as HydraNet.pth.
- train_data/val_data: Data generated according to the data generating process described in the paper, TO DO: identify the correct file associated with this simulation, and re-name things accordingly.
