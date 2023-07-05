import numpy as np
from albumentations import *
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout 

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
