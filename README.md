# Automatic-Organ-Segmentation-on-Male-Pelvis-CT

# Overview
This project focuses on automating the semantic segmentation of CT images of the male pelvis, specifically for radiotherapy treatment. In radiotherapy, it is crucial to accurately segment the prostate and surrounding organs at risk (such as the rectum and bladder) to target the radiation treatment while minimizing harm to healthy tissues. Currently, this segmentation is done manually by medical professionals, which is time-consuming and prone to inter- and intra-observer variability.

The goal of this project is to develop a neural network model that can automatically perform this segmentation, reducing both time and variability in the process.

# Methodology
To achieve this, we employed a pre-trained ResNet-18 convolutional neural network (CNN).The ResNet-18 CNN, combined with an encoder-decoder structure based on the DeepLabv3+ model, achieved accurate and efficient segmentation without the need for preprocessing techniques or extensive manual refinement Our dataset consisted of 100 CT images from patients who had undergone radiotherapy treatment. In these images, the prostate and organs at risk (rectum and bladder) were manually segmented by radiation oncologists, providing the ground truth for training the model. 







