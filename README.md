# Ant detection
____
## Introduсtion
This workspace is for building an ant recognition system using machine learning.
## Train.py
This file is for training a SSD neural network with a VGG16 pretrained backbone. To run the file you need to change some values in main function.
- set the number of epochs
- set the batch size
- specify trainig directory. This directory contains training data, which is RGB images and .xml files with labeled boundary boxes.
- specify directory where models will be saved. For each launch, in this directory will be created file with current data, which consist the best models' parameters (model, whose loss is minimum), full-trained model and loss-grafic.
## Test.py
