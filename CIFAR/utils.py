import numpy as np
from albumentations import *
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout 
from albumentations.augmentations.geometric.transforms import PadIfNeeded 
from albumentations.augmentations.crops.transforms import CenterCrop
from albumentations.augmentations.dropout.cutout import Cutout

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']
    
# Train Transformation
class TrainAlbumentation():
  def __init__(self):
    self.train_transform = Compose([
       HorizontalFlip(),
       ShiftScaleRotate(shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2), rotate_limit=(-15, 15), p=0.5),
       PadIfNeeded(min_height=36, min_width=36, pad_height_divisor=None, pad_width_divisor=None, p=1.0),
      #  CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[255*0.485,255*0.456,255*0.406], mask_fill_value = None),
       RandomCrop(32, 32, always_apply=False, p=1.0),
       CenterCrop(32, 32, always_apply=False, p=1.0),
       Cutout (num_holes=1, max_h_size=8, max_w_size=8, fill_value=[255*0.485,255*0.456,255*0.406], always_apply=False, p=0.5),
       Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
       ),
       ToTensorV2()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.train_transform(image = img)['image']
    return img
    
# Test Transformation
class TestAlbumentation():
  def __init__(self):
    self.test_transform = Compose([
       Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225],
       ),
       ToTensorV2()
    ])

  def __call__(self, img):
    img = np.array(img)
    img = self.test_transform(image = img)['image']
    return img

# To Visualize Images, n = number of images
def visualize_images(n):
  for i in range(0,n):
    plt.subplot(3, 3, i+1)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.imshow(np.transpose(((images[i]/2)+0.5).numpy(),(1,2,0)))
    plt.title(classes[labels[i]])
