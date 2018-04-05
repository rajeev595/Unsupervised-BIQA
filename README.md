# Unsupervised-BIQA
This repository contains codes to perform `Unsupervised-Blind Image Quality Assessment`.

### Files
* data_utils.py:
  This file contains the utility functions for pre-processing the data, loading batches of data while training,
  testing and evaluation.
  
* alexnet.py:
  This file creates an AlexNet graph and loads the weights that are trained on ImageNet.
  This file is ported from [alexnet-finetuning](https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d/alexnet.py)
  repository by [Frederik Kratzert](https://github.com/kratzert)
  
* Workbook.ipynb:
  This workbook does all the functionality from loading the data, creating a tensorflow session, training and evaluating the network.
  
### Dependencies
* [tensorflow](https://www.tensorflow.org/) 1.6 or later
