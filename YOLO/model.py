from torch import optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import garbage_collection_cuda
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# from utilities.utils import find_lr
from yolov3 import YOLOv3
import config
from loss import YoloLoss
# from torch_lr_finder import LRFinder
from dataset import YOLODataset
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric


def criterion(out, y):
    loss = (
            YoloLoss(out[0], y[0], config.scaled_anchors[0])
            + YoloLoss(out[1], y[1], config.scaled_anchors[1])
            + YoloLoss(out[2], y[2], config.scaled_anchors[2])
            )
    return loss

# def find_lr(model, data_loader, optimizer, criterion):
#     lr_finder = LRFinder(model, optimizer, criterion)
#     lr_finder.range_test(data_loader, end_lr=0.1, num_iter=100, step_mode='exp')
#     _, best_lr = lr_finder.plot()
#     lr_finder.reset()
#     return best_lr



class Model(LightningModule):
  def __init__(self, enable_gc='batch'):
    super(Model, self).__init__()
    self.network = YOLOv3(in_channels=3, num_classes=config.NUM_CLASSES)
    self.learning_rate = config.LEARNING_RATE
    self.enable_gc = enable_gc
    self.my_train_loss = MeanMetric()
    self.my_val_loss = MeanMetric()

    self.max_epochs = config.NUM_EPOCHS * 2 // 5
    # self.learning_rate = learning_rate
    # self.dataloader_args = dict(batch_size=128, pin_memory=True)

  def forward(self, x):
    return self.network(x)

  def common_step(self, batch, batch_ids):
    data, target = batch
    # forward pass
    outputs = self.forward(data)
    loss = criterion(outputs, target)

    # calculate training accuracy
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    acc = 100 * correct / total
    return acc, loss

  def training_step(self, batch, batch_idx):
    train_acc, loss = self.common_step(batch, batch_idx)

    # logging training loss and accuracy during training step
    self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    return loss

  def validation_step(self, batch, batch_idx):
    val_acc, loss = self.common_step(batch, batch_idx)

    # logging validation loss and accuracy during validation step
    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)#, sync_dist=True)
    self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)#, sync_dist=True)


  def configure_optimizers(self):
      optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
    #   best_lr = find_lr(self, self.train_dataloader(), optimizer, criterion)
      scheduler = optim.lr_scheduler.OneCycleLR(
          optimizer,
          max_lr=1E-3,
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

  def train_dataloader(self):
      train_dataset = YOLODataset(
        config.DATASET + '/train.csv',
        transform=config.train_transforms,
        S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
      train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
      return train_loader

  def val_dataloader(self):
        test_dataset = YOLODataset(
        config.DATASET + '/test.csv',
        transform=config.test_transforms,
        S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
        test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
        return test_loader

  def predict_dataloader(self):
      return self.val_dataloader()
  
  def on_train_batch_end(self, outputs, batch, batch_idx):
     if self.enable_gc == 'batch':
        garbage_collection_cuda()

  def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    if self.enable_gc == 'batch':
       garbage_collection_cuda()

  def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    if self.enable_gc == 'batch':
       garbage_collection_cuda()

  def on_train_epoch_end(self):
    if self.enable_gc == 'epoch':
        garbage_collection_cuda()
        print(f"Epoch: {self.current_epoch}, Global Steps: {self.global_step}, Train Loss: {self.my_train_loss.compute()}")
        self.my_train_loss.reset()

  def on_validation_epoch_end(self):
    if self.enable_gc == 'epoch':
        garbage_collection_cuda()
        print(f"Epoch: {self.current_epoch}, Global Steps: {self.global_step}, Val Loss: {self.my_val_loss.compute()}")
        self.my_val_loss.reset()

  def on_predict_epoch_end(self):
     if self.enable_gc == 'epoch':
        garbage_collection_cuda()

def main():
    num_classes = 20
    IMAGE_SIZE = 416
    INPUT_SIZE = IMAGE_SIZE  # * 2
    model = Model()
    from torchinfo import summary
    print(summary(model, input_size=(2, 3, INPUT_SIZE, INPUT_SIZE)))
    inp = torch.randn((2, 3, INPUT_SIZE, INPUT_SIZE))
    out = model(inp)
    assert out[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert out[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert out[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")


if __name__ == "__main__":
    main()