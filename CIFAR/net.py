import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(out_channels*depth, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

def norms(normtype, embedding):
  if normtype=='bn':
     print('####Batch Norm')
     return nn.BatchNorm2d(embedding)
  elif normtype=='ln':
     return nn.GroupNorm(1, embedding)
  else:
    return nn.GroupNorm(4, embedding)

class Net(nn.Module):
    def __init__(self, normtype):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 32, RF 3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 32, RF 5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 7
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 11
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 16, RF 15
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            norms(normtype, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 8, RF 19
        self.separable_conv = SeparableConv2d(
            in_channels=32, 
            out_channels=64, 
            depth=1, 
            kernel_size=(3,3)
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 10, RF 35
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            norms(normtype, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 10, RF 43
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            norms(normtype, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) # output_size = 5, RF 51
        self.gap = nn.Sequential(
            # nn.AvgPool2d(kernel_size=5)
            nn.AdaptiveAvgPool2d(1)
        ) # output_size = 1
        self.fc1 = nn.Linear(128, 10)
        #self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.convblock1(x) #3
        x = self.convblock2(x) #5
        x = self.convblock3(x) #7
        x = self.convblock4(x) #11
        x = self.convblock5(x) #15
        x = self.convblock6(x) #19
        x = self.separable_conv(x)
        # x = self.depthwise_conv #27
        # x = self.pointwise_conv #27
        x = self.convblock7(x) #35
        x = self.convblock8(x) #43
        x = self.convblock9(x) #51
        x = self.gap(x)
        x = x.view(-1, 128)
        x = (self.fc1(x))
        return x