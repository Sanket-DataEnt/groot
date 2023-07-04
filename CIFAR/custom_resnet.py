import torch.nn as nn
import torch.nn.functional as F

def normalization(norm_type, embedding):
  if norm_type=='batch':
     return nn.BatchNorm2d(embedding)
  elif norm_type=='layer':
     return nn.GroupNorm(1, embedding)
  else:
    return nn.GroupNorm(4, embedding)

def custom_conv_layer(in_channels, out_channels, kernel_size, padding, stride=1, bias=False):
  conv_layer = [
    
  ]
