import numpy as np
from albumentations import *
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout 

class TrainAlbumentation():
  def __init__(self):
    self.train_transform = Compose([
       HorizontalFlip(),
       ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
       CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[255*0.485,255*0.456,255*0.406], mask_fill_value = None),
       ShiftScaleRotate(shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2), rotate_limit=(-15, 15), p=0.5),
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