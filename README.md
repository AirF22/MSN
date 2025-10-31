# Enhancing Multi-illuminant Color Constancy through Multi-scale Illuminant Estimation and Adaptive Fusion

### About

This article is currently under consideration by The Visual Computer journal.

This is an official repository of "Enhancing Multi-illuminant Color Constancy through Multi-scale Illuminant Estimation and Adaptive Fusion".

This repository provides:

- Architecture code for the Multi-scale Net (MSN) model  
- Code for dataset loading and preprocessing
- Code for training and testing
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

   Check the Hyper Parameter and Path in hyper_parameter.yaml and run main.py

   ##### Test

   Check the Hyper Parameter and Path and run test_single.py. 

   

2. ##### The Cube++ dataset can be download from [here](https://github.com/Visillect/CubePlusPlus)

   ##### Train

   Check the Hyper Parameter and Path in hyper_parameter.yaml and run main.py

   ⚠**For single-illuminant images, we calculate the mean values of both the model's predicted illuminant map and the ground-truth illuminant map, then use these values as error statistics for evaluation and testing purposes.**

   ⚠⚠**In the Cube++ dataset, no mask file is needed because the SpyderCube is already cropped out.**

   ##### Test

   Check the Hyper Parameter and Path and run test_single.py. 

   

3. **Pre-trained models can be downloaded from [here](https://pan.baidu.com/s/1xWsgbX58t9H-RhLChcZwKQ?pwd=j946)**

   

   
