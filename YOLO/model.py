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
# from torchmetrics import MeanMetric
from torchmetrics import Accuracy
from utils import plot_couple_examples, check_class_accuracy, save_checkpoint, get_evaluation_bboxes, mean_average_precision

class Model(LightningModule):
  def __init__(self, enable_gc='batch'):
    super(Model, self).__init__()
    self.model = YOLOv3(in_channels=3, num_classes=config.NUM_CLASSES)
    self.learning_rate = config.LEARNING_RATE
    self.accuracy = Accuracy('MULTICLASS', num_classes=20)
    self.loss = YoloLoss()
    self.train_transform = config.tarain_transforms
    self.test_transform = config.test_transforms

    self.max_epochs = config.NUM_EPOCHS * 2 // 5

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    y0, y1, y2 = (
        y[0],
        y[1],
        y[2],
    )
    with torch.cuda.amp.autocast():
        out = self(x)
        loss = (
            self.loss(out[0], y0, config.scaled_anchors[0])
            + self.loss(out[1], y1, config.scaled_anchors[1])
            + self.loss(out[2], y2, config.scaled_anchors[2])
        )
    self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y0, y1, y2 = y[0], y[1], y[2]

    with torch.cuda.amp.autocast():
        out = self(x)
        loss = (
            self.loss(out[0], y0, config.scaled_anchors[0])
            + self.loss(out[1], y1, config.scaled_anchors[1])
            + self.loss(out[2], y2, config.scaled_anchors[2])
        )

    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    garbage_collection_cuda()
    return loss
  
  def on_validation_epoch_end(self):
    plot_couple_examples(self.model, self.test_dataloader(), 0.6, 0.5, config.scaled_anchors)
    # Get the learning rate from the optimizer
    optimizer = self.optimizers()
    current_learning_rate = optimizer.param_groups[0]['lr']
    print("Current learning rate: "+str(current_learning_rate))
    epoch = self.current_epoch
    print(f"Currently epoch {epoch}")
    print("On Train Eval loader:")
    print("On Train loader:")
    check_class_accuracy(self.model, self.train_dataloader(), threshold=config.CONF_THRESHOLD)
    epoch = self.current_epoch
    if config.SAVE_MODEL:
        save_checkpoint(self.model, optimizer, filename="checkpoints/"+str(epoch)+f"checkpoint.pth.tar")

    if epoch > 0 and epoch % 10 == 0:
        check_class_accuracy(self.model, self.test_dataloader(), threshold=config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_evaluation_bboxes(
            self.test_dataloader(),
            self.model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
        )
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"MAP: {mapval.item()}")
    garbage_collection_cuda()

  def on_epoch_end(self):
    garbage_collection_cuda()

  def test_step(self, batch, batch_idx):
    # Here we just reuse the validation_step for testing
    return self.validation_step(batch, batch_idx)


def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    EPOCHS = config.NUM_EPOCHS // 5
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1E-3,
        steps_per_epoch=len(self.train_dataloader()),
        total_steps = 41,
        epochs=EPOCHS,
        pct_start=0.2,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Example StepLR configuration

    return [optimizer], [scheduler]

####################
# DATA RELATED HOOKS
####################
def setup(self, stage=None):
    self.train_dataset = YOLODataset(
        config.DATASET + "/train.csv",
        transform=self.train_transform,
        S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        )
    self.test_dataset = YOLODataset(
        config.DATASET + "/test.csv",
        transform=self.test_transform,
        S=[config.IMAGE_SIZE // 32, config.IMAGE_SIZE // 16, config.IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        )

def train_dataloader(self):

    return DataLoader(self.train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

def val_dataloader(self):

    return DataLoader(self.test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

def test_dataloader(self):

    return DataLoader(self.test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

def get_optimizer(self):
    return self.optimizers()
  

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