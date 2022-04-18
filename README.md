# Ant detection
## Introduсtion
This workspace is for building an ant recognition system using machine learning.
## Train.py
This file is for training a SSD neural network with a VGG16 pretrained backbone. Run script in terminal and pass the parameters *in the following order*:
- *Number of epochs.*
- *Batch size.*
- *Training directory.* This directory contains training data, which is RGB images and .xml files with labeled bounding boxes.
- *Model directory.* For each launch, in this directory will be created file with current data, which consist the best models' parameters (model, whose loss is minimum), full-trained model and loss-graph.
## Test.py
This file is for testing trained model. Returns an image with predicted and true bounding boxes. Run script in terminal and pass the parameters *in the following order*:
- *Load path.* Full path of model to load and test.
- *Image path.* Full path of image on which the test will be.
