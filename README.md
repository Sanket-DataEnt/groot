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
