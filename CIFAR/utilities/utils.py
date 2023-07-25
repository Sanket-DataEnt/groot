import numpy as np
import torch
from tqdm import tqdm
from albumentations import *
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout 
from albumentations.augmentations.geometric.transforms import PadIfNeeded 
from albumentations.augmentations.crops.transforms import CenterCrop
from albumentations.augmentations.dropout.cutout import Cutout
import matplotlib.pyplot as plt

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
def visualize_images(n, images, labels, classes):
  for i in range(0,n):
    plt.subplot(3, 3, i+1)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.imshow(np.transpose(((images[i]/2)+0.5).numpy(),(1,2,0)))
    plt.title(classes[labels[i]])

train_losses = []
train_acc = []

# Training Function
def train(net, device, trainloader, optimizer, criterion, epoch):
  net.train()
  pbar = tqdm(trainloader)
  running_loss = 0.0
  for i, (data, target) in enumerate(pbar):
        # get the inputs
        correct = 0
        processed = 0
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # Predict
        y_pred = net(data)
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        #pbar.set_description(desc= f'Epoch= {epoch} Loss={loss.item()} Batch_id={i} Accuracy={100*correct/processed:0.2f}')
        pbar.update(1)
  train_acc = 100*correct/processed
  print(f'Epoch= {epoch} Loss={loss.item()} Accuracy={100*correct/processed:0.2f}')
  return train_acc

def test(net, device, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar1 = tqdm(testloader)
        for i, (data, target) in enumerate(pbar1):
           data, target = data.to(device), target.to(device)
           outputs = net(data)
           _, predicted = torch.max(outputs.data, 1)
           total += target.size(0)
           correct += (predicted == target).sum().item()
          
        print('Accuracy of the network on the 10000 test images: %0.2f %%' % (100 * correct / total))
    test_acc = (100 * correct / total)  

    return test_acc

def test_categorywise(net, device, testloader, classes):
   class_correct = list(0. for i in range(10))
   class_total = list(0. for i in range(10))
   with torch.no_grad():
      pbar = tqdm(testloader)
      for i, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        outputs = net(data)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == target).squeeze()
        for i in range(4):
            label = target[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


   for i in range(10):
      print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def plots(train_acc, test_acc, learning_rate):
  fig, axs = plt.subplots(3)
  axs[0].plot(train_acc)
  axs[0].set_title("Training Accuracy")
  axs[0].set_xlabel("Batch")
  axs[0].set_ylabel("Accuracy")
  axs[1].plot(test_acc)
  axs[1].set_title("Test Accuracy")
  axs[1].set_xlabel("Batch")
  axs[2].plot(learning_rate)
  axs[2].set_title("Learning rate")
  axs[2].set_xlabel("epoch")
  axs[2].set_ylabel("lr")
