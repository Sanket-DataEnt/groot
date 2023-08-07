from torch import nn
from torch import optim
from pytorch_lightning import LightningModule
import torch
import torchvision
from utilities.utils import TrainAlbumentation
from utilities.utils import TestAlbumentation
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
    normalization(norm_type = 'batch', embedding = out_channels),
  )
  conv_layer.append(
    nn.ReLU()
  )
  block = nn.Sequential(*conv_layer)
  return block


class Model(LightningModule):
  def __init__(self, max_epochs=24, learning_rate = 1e-7):
    super(Model, self).__init__()
    self.criterion = nn.CrossEntropyLoss()
    self.max_epochs = max_epochs
    self.learning_rate = learning_rate
    self.dataloader_args = dict(batch_size=128, pin_memory=True)

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
    return F.softmax(x, dim=-1)

  def commom_step(self, batch, batch_ids):
    data, target = batch
    # forward pass
    outputs = self.forward(data)
    loss = self.criterion(outputs, target)

    # calculate training accuracy
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    acc = 100 * correct / total
    return acc, loss

  def training_step(self, batch, batch_idx):
    train_acc, loss = self.commom_step(batch, batch_idx)

    # logging training loss and accuracy during training step
    self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    return loss

  def validation_step(self, batch, batch_idx):
    val_acc, loss = self.commom_step(batch, batch_idx)

    # logging validation loss and accuracy during validation step
    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)#, sync_dist=True)
    self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)#, sync_dist=True)


  def configure_optimizers(self):
      optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
      best_lr = find_lr(self, self.train_dataloader(), optimizer, self.criterion)
      scheduler = optim.lr_scheduler.OneCycleLR(
          optimizer,
          max_lr=best_lr,
          steps_per_epoch=len(self.train_dataloader()),
          epochs=self.max_epochs,
          pct_start=5/self.max_epochs,
          div_factor=100,
          three_phase=False,
          final_div_factor=100,
          anneal_strategy='linear'
      )
      return {
          'optimizer': optimizer,
          'lr_scheduler': {
              "scheduler": scheduler,
              "interval": "step",
          }
      }

  def prepare_data(self):
      torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
      torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

  def setup(self, stage=None):
      if stage == 'fit' or stage is None:
          self.cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, transform=TrainAlbumentation())
          self.cifar10_val = torchvision.datasets.CIFAR10(root='./data', train=False, transform=TestAlbumentation())
      if stage == 'test' or stage is None:
          self.cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=TestAlbumentation())

  def train_dataloader(self):
      return torch.utils.data.DataLoader(self.cifar10_train, shuffle=True, **self.dataloader_args)

  def val_dataloader(self):
      return torch.utils.data.DataLoader(self.cifar10_val, shuffle=False, **self.dataloader_args)

  def predict_dataloader(self):
      return self.val_dataloader()