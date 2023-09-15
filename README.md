# Valify
## Author
Ahmed Hussein
## table of Content 
1. [Install for Train Enviroments](#install-for-train-enviroments)
2. [Model Information](#model-information)
3. [Dataset](#dataset)
4. [Train](#train)
5. [Metric](#metric)
6. [Inference](#inference)
7. [Docker Setup](#docker-setup)
8. [Test API](#test-api)

## Install for Train Enviroments
to install you need to create Virtual Environment with conda, activate it  then install requirments
```bash
conda create -n Test python=3.8.*
conda activate Test
pip install -r requirements.txt
```
## Model Information
the model informations
### Architecture

```
========== Model Summary  ==========
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 16, 16]             224
            Conv2d-2             [-1, 16, 8, 8]           1,168
            Conv2d-3             [-1, 32, 4, 4]           4,640
            Linear-4                  [-1, 128]          65,664
            Linear-5                  [-1, 256]          33,024
            Linear-6                   [-1, 29]           7,453
================================================================
Total params: 112,173
Trainable params: 112,173
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.03
Params size (MB): 0.43
Estimated Total Size (MB): 0.47
```

### MACs 
you can compute FLOPs (Floating Point Operations) From MACs as one MACs equals roughly two FLOPs
```
MyNetwork(
  112.17 k, 100.000% Params, 312.48 KMac, 100.000% MACs, 
  (conv1): Conv2d(224, 0.200% Params, 57.34 KMac, 18.351% MACs, 3, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (conv2): Conv2d(1.17 k, 1.041% Params, 74.75 KMac, 23.922% MACs, 8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (conv3): Conv2d(4.64 k, 4.136% Params, 74.24 KMac, 23.759% MACs, 16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (fc1): Linear(65.66 k, 58.538% Params, 65.66 KMac, 21.014% MACs, in_features=512, out_features=128, bias=True)
  (fc2): Linear(33.02 k, 29.440% Params, 33.02 KMac, 10.568% MACs, in_features=128, out_features=256, bias=True)
  (fc3): Linear(7.45 k, 6.644% Params, 7.45 KMac, 2.385% MACs, in_features=256, out_features=29, bias=True)
)
========== Model Complex  ==========
Computational complexity:       312.48 KMac
Number of parameters:           112.17 k
```
### Respective Field
the respective field represent the number of pixels the model see in as overall in a CNN  for conv layers
you can see this [link](https://rubikscode.net/2021/11/15/receptive-field-arithmetic-for-convolutional-neural-networks/)
```
input_layer: output size = 32; size change relative to original = 1; receptive image size = 1
Conv2d-1: output size = 16; size change relative to original = 2; receptive image size = 3
Conv2d-2: output size = 8; size change relative to original = 4; receptive image size = 7
Conv2d-3: output size = 4; size change relative to original = 8; receptive image size = 15
```

## Dataset
the dataset contain Train and Dev data 
### Hierarchy  
```tree
Dataset
|
----Train 
    |
    ----29 folder each folder is an arabic letter, each folder contain 2385 image 
|
----Dev 
    |
    ----29 folder each folder is an arabic letter, each folder contain 96 image
```
### Dataset location
you can find the dataset in [link](Dataset/Link.txt)

## Train
train 
* **LR** : Start with 1e-3 then end with 0.5*1e-3 through the training 
* **Epochs** : 9 epochs
* **Loss** : NLLLoss  The negative log likelihood loss, Calculated as −log(y), where y is a prediction corresponding to the true label, after the Softmax Activation
* **Log** : logs can be found in [tensorboard](runs/Sep13_22-44-54_res12-Precision-T7600/events.out.tfevents.1694637894.res12-Precision-T7600.1899822.0)
you can run 
```bash
tensorboard --logdir=runs
```
you can see the [screenshots](Images) 
