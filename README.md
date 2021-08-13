# Simultaneous-Road-Network-Segmentation-and-Vectorization

We present a new automatic deep learning-based network, i.e., a Road Vectorization Network (RoadVecNet) that comprises interlinked convolutional UNet networks to tackle road
segmentation and road vectorization challenges at the same time. Particularly, RoadVecNet contains two UNet networks. The first network with powerful representation capability
is capable of obtaining more coherent and satisfactory road segmentation maps even under complex urban setup, and the second network is linked to the first network to vectorize
road networks by making whole utilization of feature maps generated previously. We utilize a loss function called Focal loss weighted by median frequency balancing (MFB FL) to
focus more on the hard samples, fix the training data imbalance problem, and improve the road extraction and vectorization performance. Also, a new module named Dense dilated
spatial pyramid pooling (DDSPP), which combines the benefit of cascaded modules with atrous convolution and Atrous spatial pyramid pooling (ASPP) is designed to produce more
scale features over a broader range. Two types of high-resolution remote sensing datasets, i.e., Aerial and Google Earth imagery, were used for both road segmentation and road
vectorization tasks.

Prerequisites and Run

This code has been implemented in python language using Keras libarary with tensorflow backend and tested, though should be compatible with related environment. following Environement and Library needed to run the code:

Python 3 over
Keras - tensorflow backend

Run Demo

For training deep models, go to the related folder and follow the bellow steps:

i) Download the Massachusets and Ottawa road datasets from this link https://www.cs.toronto.edu/~vmnih/data/ and create both training dataset and ground truth dataset.

ii) Run Prepare_dataset.py for data preparation and dividing data to train, validation and test sets.

iii) Run models.py for training RoadVecNet model using training and validation sets.

iv) For performance calculation and producing segmentation result, run evaluate.py.

Quick Overview
