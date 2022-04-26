# Ant detection
## Introduсtion
This workspace is for building an ant recognition system using machine learning.
## model.py
This file contain the architecture of the model based on MobileNetV3 large.
## model_train.py
This file is for training a SSD neural network with a VGG16 pretrained backbone. Run script in terminal and pass the parameters **in the following order**:
- *Number of epochs.* Type=int.
- *Batch size.* Type=int.
- *Training directory.* This directory contains training data, which is RGB images and .xml files with labeled bounding boxes. Type=str.
- *Model directory.* For each launch, in this directory will be created file with current data, which consist the best models' parameters (model, whose loss is minimum), full-trained model and loss-graph. Type=str.
- *Learning rate.* Enter learning rate for optimizer (Adam using now).Type=float.
- *Maximum objects.* Enter maximum number of objects detected per image. Type=int.
## model_test.py
:negative_squared_cross_mark: in progress :negative_squared_cross_mark:
This file is for testing trained model. Returns an image with predicted and true bounding boxes. Run script in terminal and pass the parameters **in the following order**:
- *Load path.* Full path of model to load and test.
- *Image path.* Full path of image on which the test will be.
