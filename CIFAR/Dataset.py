from torchvision import datasets, transforms
import torch
import torchvision
import torchvision.transforms as transforms
from utils import TrainAlbumentation
from utils import TestAlbumentation

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=32, pin_memory=True) if torch.cuda.is_available() else dict(shuffle=True, batch_size=64)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=TrainAlbumentation())
trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=TestAlbumentation())
testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
