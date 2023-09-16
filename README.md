# Valify
## Author
### Ahmed Hussein
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
the dataset contain Train and Dev data can be generated from [general_functions.py](utils/general_functions.py)
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
train can be found in [train.py](train.py)
* **Input Size** : 32 width, 32 height and BGR
* **LR** : Start with 1e-3 then end with 0.5*1e-3 through the training 
* **Epochs** : 9 epochs
* **Loss** : NLLLoss  The negative log likelihood loss, Calculated as −log(y), where y is a prediction corresponding to the true label, after the Softmax Activation
* **Log** : logs can be found in [tensorboard](runs/Sep13_22-44-54_res12-Precision-T7600/events.out.tfevents.1694637894.res12-Precision-T7600.1899822.0)
you can run the next command or you can see the [screenshots](Images) 
```bash
tensorboard --logdir=runs
```

run the train with 
```bash
python train.py
```

## Metric
the metric is **Precision**, **Recall**, and **F1 Score**
you can generate it from the [get_metrics.py](get_metrics.py)
```bash
python get_metrics.py
```
```json
{"ا": {"Precision": 1.0, "Recall": 0.9479166666666666, "F1": 0.9732620320855615}, "ب": {"Precision": 0.8055555555555556, "Recall": 0.90625, "F1": 0.8529411764705882}, "ت": {"Precision": 0.7373737373737373, "Recall": 0.7604166666666666, "F1": 0.7487179487179487}, "ث": {"Precision": 0.7604166666666666, "Recall": 0.7604166666666666, "F1": 0.7604166666666666}, "ج": {"Precision": 0.971830985915493, "Recall": 0.71875, "F1": 0.8263473053892216}, "ح": {"Precision": 0.8640776699029126, "Recall": 0.9270833333333334, "F1": 0.8944723618090452}, "خ": {"Precision": 0.9263157894736842, "Recall": 0.9166666666666666, "F1": 0.9214659685863874}, "د": {"Precision": 0.8942307692307693, "Recall": 0.96875, "F1": 0.93}, "ذ": {"Precision": 0.9368421052631579, "Recall": 0.9270833333333334, "F1": 0.9319371727748692}, "ر": {"Precision": 0.93, "Recall": 0.96875, "F1": 0.9489795918367346}, "ز": {"Precision": 0.9891304347826086, "Recall": 0.9479166666666666, "F1": 0.9680851063829786}, "س": {"Precision": 0.85, "Recall": 0.8854166666666666, "F1": 0.8673469387755102}, "ش": {"Precision": 0.956989247311828, "Recall": 0.9270833333333334, "F1": 0.9417989417989417}, "ص": {"Precision": 0.8105263157894737, "Recall": 0.8020833333333334, "F1": 0.806282722513089}, "ض": {"Precision": 0.946236559139785, "Recall": 0.9166666666666666, "F1": 0.9312169312169313}, "ط": {"Precision": 0.9775280898876404, "Recall": 0.90625, "F1": 0.9405405405405405}, "ظ": {"Precision": 0.9108910891089109, "Recall": 0.9583333333333334, "F1": 0.934010152284264}, "ع": {"Precision": 0.7924528301886793, "Recall": 0.875, "F1": 0.8316831683168316}, "غ": {"Precision": 0.7435897435897436, "Recall": 0.90625, "F1": 0.8169014084507042}, "ف": {"Precision": 0.7560975609756098, "Recall": 0.6458333333333334, "F1": 0.6966292134831461}, "ق": {"Precision": 0.7934782608695652, "Recall": 0.7604166666666666, "F1": 0.7765957446808509}, "ك": {"Precision": 0.9777777777777777, "Recall": 0.9166666666666666, "F1": 0.946236559139785}, "ل": {"Precision": 0.8529411764705882, "Recall": 0.90625, "F1": 0.8787878787878787}, "لا": {"Precision": 0.979381443298969, "Recall": 0.9895833333333334, "F1": 0.9844559585492227}, "م": {"Precision": 0.8736842105263158, "Recall": 0.8645833333333334, "F1": 0.8691099476439791}, "ن": {"Precision": 0.8395061728395061, "Recall": 0.7083333333333334, "F1": 0.768361581920904}, "ه": {"Precision": 0.7889908256880734, "Recall": 0.8958333333333334, "F1": 0.8390243902439025}, "و": {"Precision": 0.9894736842105263, "Recall": 0.9791666666666666, "F1": 0.9842931937172775}, "ي": {"Precision": 0.9139784946236559, "Recall": 0.8854166666666666, "F1": 0.8994708994708994}}
```
## Inference
the inference can be found in [inference.py](inference.py) 
run inference with 
```bash
python inference.py <dir-contain-images>
```
this will print the predicted label 

## Docker Setup
you can setup the docker from zero from (Dockerfile)[API/Dockerfile]
run command 
```bash
sudo docker build -t app .
```
or load it 
