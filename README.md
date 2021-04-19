# GenericFramework

## Introduction
This branch is the implementation of noised differentially private machine learning tasks on TF2 :smile:.

The current framework is run with the environment:

* Tensorflow == 2.4.1
* Numpy == 1.19.5
* Scipy == 1.5.3 (If IDN noise mechanism is implemented)
* Cvxpy == 1.1.12 (If MM noise mechanism is implemented)

## Utility Modules
Let's see the utility modules first. 

* `data_generator.py`  A dataset handler. Currently supporting *MNIST*,*CIFAR-10*,(data from keras.dataset) *SVHN*,  *Adult*(data needs to be downloaded from the website and put in `\data`). Custom dataset is supported and more datasets are waiting to be added.
* `network.py` A network model handler. Currently supporting *LeNet*, *AlexNet*, *VGG-16* and the *FCNet*(specific for *Adult*), Custom network is supported.
* `special_printer` Generates colored printouts. Has functions like `info`, `heading`, `news` and `warning`. Once a log is specified, all `special_printer` handled printouts will be recorded to the log file. (Credit to @ashleyxly)
* `generate_argparse.py` Takes a `dictionary` and generates command line options. Saves developers the hassle of writing `argparse` code. (Credit to @ashleyxly)

## Noise Generate Module
The `noise_gen_wrapper.py`, `noise_generator.py` and `noise_utils\.` belong to noise generate module.

1) `noise_gen_wrapper.py` is the interface between the caller from `main.py` and `noise_generator.py`. It reads the `net_info` to initiate different type of noise generation mechanisms.

2) `noise_generator.py` is called by `noise_gen_wrapper.py` to initiate corresponding mechanisms and generate the noise.
* Initialization: takes in python or numpy type values to calculate parameters.
* Generation: takes in a tensor type gradients (no need to be flattened), outputs a tensor type noise of the same shape as the input.

3) `noise_utils\.` contains the implementations of several noise mechanisms to generate the actual noise, including
* Gaussian Mechanism (Martin Abadi, 2016)
* Optimized Mechanism (Liyao Xiang, 2018)
* MVG Mechanism (Thee Chanyaswad, 2018)
* IMGM Mechanism (Unpublished)
* UDN, IDN Mechanism (Unpublished)
* MM Mechanism (Chao Li, 2014) (Implementation sepcific for Adult dataset, or it is too slow)

## Core Modules 
The most important file is `main.py`(`main_clip_new` differs only in the way of clipping).

Users can custom the parameters in `net_info`. We custom the training process based on the model.fit of keras and custom the feedbacks. We calculate per example gradients and accelerate the operation using the vectorized way in TF2. We achieve the accumulating gradients operation(update the model a lot, several batches) since the per example gradients occupy the memory and this operation can solve the OOM errors.