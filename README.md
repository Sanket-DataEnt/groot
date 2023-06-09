# groot
This repo contains detailed implementation of Computer Vision problems

## Details of files used in this repository

### mnist_training

- `This file is performing a model training for MNIST dataset` 

### models

- `This file contains different models for the experimentation`
- **model 1 summary :-**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

- **model 2 summary :-**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
            Conv2d-2           [-1, 64, 24, 24]          18,432
            Conv2d-3          [-1, 128, 10, 10]          73,728
            Conv2d-4            [-1, 256, 8, 8]         294,912
            Linear-5                   [-1, 50]         204,800
            Linear-6                   [-1, 10]             500
================================================================
Total params: 592,660
Trainable params: 592,660
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.93
----------------------------------------------------------------
```

### utils

- `This file contains following utilities functions`
    - `GetCorrectPredCount` : To predict the correct predicted values
    - `train` : Function to perform the training
    - `test` : Function to do testing of the model
    - `plot` : Function to plot the accuracy and loss for training and test

### S6

 `This folder contains script to achieve 99.41% accuracy with approx 16K parameters in 20 epochs on MNIST dataset`

 - **model summary :-**

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
         Dropout2d-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,640
              ReLU-6           [-1, 32, 24, 24]               0
       BatchNorm2d-7           [-1, 32, 24, 24]              64
         Dropout2d-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 16, 24, 24]             528
        MaxPool2d-10           [-1, 16, 12, 12]               0
           Conv2d-11           [-1, 32, 10, 10]           4,640
             ReLU-12           [-1, 32, 10, 10]               0
      BatchNorm2d-13           [-1, 32, 10, 10]              64
        Dropout2d-14           [-1, 32, 10, 10]               0
           Conv2d-15           [-1, 16, 10, 10]             528
           Conv2d-16             [-1, 32, 8, 8]           4,640
             ReLU-17             [-1, 32, 8, 8]               0
      BatchNorm2d-18             [-1, 32, 8, 8]              64
        Dropout2d-19             [-1, 32, 8, 8]               0
           Conv2d-20              [-1, 8, 8, 8]             264
           Conv2d-21             [-1, 16, 6, 6]           1,168
             ReLU-22             [-1, 16, 6, 6]               0
      BatchNorm2d-23             [-1, 16, 6, 6]              32
        Dropout2d-24             [-1, 16, 6, 6]               0
AdaptiveAvgPool2d-25             [-1, 16, 1, 1]               0
           Linear-26                   [-1, 10]             170
================================================================
Total params: 16,994
Trainable params: 16,994
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.17
Params size (MB): 0.06
Estimated Total Size (MB): 1.24
----------------------------------------------------------------
```
