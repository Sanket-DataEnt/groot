import torch.nn as nn
import torch.nn.functional as F

def normalization(norm_type, embedding):
  if norm_type=='batch':
     return nn.BatchNorm2d(embedding)
  elif norm_type=='layer':
     return nn.GroupNorm(1, embedding)
  else:
    return nn.GroupNorm(4, embedding)

def custom_conv_layer(in_channels, 
                      out_channels, 
                      pool,
                      norm_type,
                      ):
  conv_layer = [
     nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, bias=False)
  ]
  if pool :
    conv_layer.append(
      nn.MaxPool2d(2, 2),
    )
  conv_layer.append(
    normalization(norm_type, out_channels),
  )
  conv_layer.append(
    nn.ReLU()
  )
  block = nn.Sequential(*conv_layer)
  return block

class Net(nn.Module):
  def __init__(self, normtype):
      super(Net, self).__init__()
      # prep layer - PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
      self.prep_layer = custom_conv_layer(3, 64, False, 'batch')
      # Layer1 -
      #   X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
      #   R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
      #   Add(X, R1)
      self.layer1_x = custom_conv_layer(64, 128, True, 'batch')
      self.layer1_r1 = nn.Sequential(
        custom_conv_layer(128, 128, False, 'batch')
        custom_conv_layer(128, 128, False, 'batch')
      )
      # Layer 2 -
          # Conv 3x3 [256k]
          # MaxPooling2D
          # BN
          # ReLU
      self.layer2 = custom_conv_layer(128, 256, True, 'batch')
      # Layer 3 -
      #   X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
      #   R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
      #   Add(X, R2)
      self.layer3_x = custom_conv_layer(256, 512, True, 'batch')




