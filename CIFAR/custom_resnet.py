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
      # prep layer
      self.prep_layer = custom_conv_layer(3, 64, False, 'batch')
      # layer 1
      self.layer1_x = custom_conv_layer(64, 128, True, 'batch')
      self.layer1_r1 = nn.Sequential(
        custom_conv_layer(128, 128, False, 'batch'),
        custom_conv_layer(128, 128, False, 'batch')
      )
      # layer 2
      self.layer2 = custom_conv_layer(128, 256, True, 'batch')
      # Layer 3
      self.layer3_x = custom_conv_layer(256, 512, True, 'batch')
      self.layer3_r3 = nn.Sequential(
        custom_conv_layer(512, 512, False, 'batch'),
        custom_conv_layer(512, 512, False, 'batch')       
      )
      # MaxPooling with Kernel Size 4
      self.pool = nn.MaxPool2d(4, 4)
      # FC Layer 
      self.fc = nn.Linear(512, 10)

  def forward(self, x):
    x = self.prep_layer(x)
    x1 = self.layer1_x(x)
    r1 = self.layer1_r1(x1)
    x = x1 + r1
    x = self.layer2(x)
    x3 = self.layer3_x(x)
    r3 = self.layer3_r3(x3)
    x = x3 + r3
    x = self.pool(x)
    x = x.view(-1, 512)
    x = self.fc(x)
    return x
