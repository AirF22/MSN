# Multi-illuminant Color Constancy via Multi-scale  Illuminant Estimation and Fusion

### About

This article is currently under consideration by The Visual Computer journal.

This is an official repository of "Multi-illuminant Color Constancy via Multi-scale  Illuminant Estimation and Fusion".

This repository provides:

- Architecture code for the Multi-scale Net (MSN) model
  
- Code for dataset loading and preprocessing
- Code for training and testing
- Code for statistics-based methods implemented by python
- Code for white balance and visualization
- Pre-trained model parameter file for Multi-scale Net



### Requirements

Our running environment is as follows:

- Python version 3.12.4

- Pytorch version 2.4.0
- CUDA version 12.6



### Dataset and Usage

No new datasets were created in our paper, all existing publicly available datasets were used. The public datasets can be found in the following links, and the preprocessing methods are described in the paper.

1. ##### The LSMI dataset can be download from [here](https://github.com/DY112/LSMI-dataset?tab=readme-ov-file)

   ##### Train

   Check the Hyper Parameter and Path in train.py and run train.py

   ##### Test

   Check the Hyper Parameter and Path in test.py and run test.py. 

   

2. ##### The Cube++ dataset can be download from [here](https://github.com/Visillect/CubePlusPlus)

   ##### Train

   Check the Hyper Parameter and Path in train.py and run train.py

   ⚠**For single-illuminant images, we calculate the mean values of both the model's predicted illuminant map and the ground-truth illuminant map, then use these values as error statistics for evaluation and testing purposes.**

   ⚠⚠**The cube++ dataset does not need to load the mask file since the Spyder Cube is removed at the time of cropping**

   ##### Test

   Check the Hyper Parameter and Path in test.py and run test.py. 

   

3. **Pre-trained models can be downloaded from [here](https://pan.baidu.com/s/1xWsgbX58t9H-RhLChcZwKQ?pwd=j946)**

   

   
