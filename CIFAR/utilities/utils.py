import numpy as np
import torch
import math
from tqdm import tqdm
from torchvision import transforms
from albumentations import *
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout 
from albumentations.augmentations.geometric.transforms import PadIfNeeded 
from albumentations.augmentations.crops.transforms import CenterCrop
from albumentations.augmentations.dropout.cutout import Cutout
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch_lr_finder import LRFinder


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")

def find_lr(model, data_loader, optimizer, criterion):
    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(data_loader, end_lr=0.1, num_iter=100, step_mode='exp')
    _, best_lr = lr_finder.plot()
    lr_finder.reset()
    return best_lr
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

def show_images(net, testloader, device, classes, flag):
  net.eval()
  missed = []
  pred = []
  targ = []
  empty_tensor = torch.tensor([]).to(device)
  with torch.no_grad():
      pbar1 = tqdm(testloader)
      for i, (data, target) in enumerate(pbar1):
           data, target = data.to(device), target.to(device)
           outputs = net(data)
           _, predicted = torch.max(outputs.data, 1)
           target1 = target.cpu().numpy()
           predicted1 = predicted.cpu().numpy()
           for i in range(64):
             if flag==1:
              if target1[i]==predicted1[i]:
                 missed.append(i)
                 new_tensor = data[i].unsqueeze(0)
                 empty_tensor = torch.cat((empty_tensor, new_tensor), dim=0)
                 pred.append(predicted1[i])
                 targ.append(target1[i])
             else:
              if target1[i]!=predicted1[i]:
                 missed.append(i)
                 new_tensor = data[i].unsqueeze(0)
                 empty_tensor = torch.cat((empty_tensor, new_tensor), dim=0)
                 pred.append(predicted1[i])
                 targ.append(target1[i])
           break

  plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
  for i in range(0,10):
   plt.subplot(5, 2, i+1)
   frame1 = plt.gca()
   frame1.axes.xaxis.set_ticklabels([])
   frame1.axes.yaxis.set_ticklabels([])
   plt.imshow(np.transpose(((data[missed[i]].cpu()/2)+0.5).numpy(),(1,2,0)))
   plt.ylabel("GT:"+str(classes[target1[missed[i]]])+'\nPred:'+str(classes[predicted1[missed[i]]]))
  return empty_tensor, pred, targ

def grad_cam(net, img, targ, image_tensor, target_layers):
  # target_layers = [net.layer4[-1]]
  # Construct the CAM object once, and then re-use it on many images:
  cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
  # We have to specify the target we want to generate
  # the Class Activation Maps for.
  # If targets is None, the highest scoring category
  # will be used for every image in the batch.
  # Here we use ClassifierOutputTarget, but you can define your own custom targets
  # That are, for example, combinations of categories, or specific outputs in a non standard model.
  targets = [ClassifierOutputTarget(targ[img])]

  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
  grayscale_cam = cam(input_tensor=image_tensor[img].unsqueeze(0), targets=targets)

  # In this example grayscale_cam has only one image in the batch:
  grayscale_cam = grayscale_cam[0, :]

  input_image = image_tensor[img].permute(1, 2, 0)
  input_image_normalized = (input_image - input_image.min()) / (input_image.max() - input_image.min())

  # Convert the normalized image tensor to NumPy array and change data type to np.float32
  input_image_np = input_image_normalized.cpu().numpy().astype(np.float32)

  visualization = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)
  return input_image_np, visualization

def visualize_gradcam(net, image_tensor, targ, pred, classes, target_layers):
  plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.2)
  maxi = 10
  # Set the figure size to adjust the size of the displayed images
  plt.figure(figsize=(10, 2 * maxi))  # Adjust the width and height as needed
  for img in range(maxi):
    input_image_np,visualization=grad_cam(net, img, targ, image_tensor, target_layers)
    plt.subplot(maxi,2,2*img+1)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.imshow(input_image_np)
    plt.subplot(maxi,2,2*img+2)
    plt.ylabel("GT:"+str(classes[targ[img]])+'\nPred:'+str(classes[pred[img]]))
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.imshow(visualization)
    
def get_misclassified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()
    # List to store misclassified Images
    misclassified_data = []
    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:
            # Migrate the data to the device
            data, target = data.to(device), target.to(device)
            # Extract single image, label from the batch
            for image, label in zip(data, target):
                # Add batch dimension to the image
                image = image.unsqueeze(0)
                # Get the model prediction on the image
                output = model(image)
                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)
                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data

def display_gradcam_output(data: list,
                           classes: list[str],
                           inv_normalize: transforms.Normalize,
                           model: 'DL Model',
                           target_layers: list['model_layer'],
                           targets=None,
                           number_of_samples: int = 10,
                           transparency: float = 0.60):
    """
    Function to visualize GradCam output on the data
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param model: Model architecture
    :param target_layers: Layers on which GradCam should be executed
    :param targets: Classes to be focused on for GradCam
    :param number_of_samples: Number of images to print
    :param transparency: Weight of Normal image when mixed with activations
    """
    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    # Create an object for GradCam
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # Iterate over number of specified images
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Get back the original image
        img = input_tensor.squeeze(0).to('cpu')
        img = inv_normalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()

        # Mix the activations on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)

        # Display the images on the plot
        plt.imshow(visualization)
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])

def get_misclassified_data_lightning(model, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()
    # List to store misclassified Images
    misclassified_data = []
    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:
            # Migrate the data to the device
            # data, target = data.to(device), target.to(device)
            # Extract single image, label from the batch
            for image, label in zip(data, target):
                # Add batch dimension to the image
                image = image.unsqueeze(0)
                # Get the model prediction on the image
                output = model(image)
                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)
                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data

def show_images_lightning(net, testloader, classes, flag):
  net.eval()
  missed = []
  pred = []
  targ = []
  empty_tensor = torch.tensor([])#.to(device)
  with torch.no_grad():
      pbar1 = tqdm(testloader)
      for i, (data, target) in enumerate(pbar1):
          #  data, target = data.to(device), target.to(device)
           outputs = net(data)
           _, predicted = torch.max(outputs.data, 1)
           target1 = target.cpu().numpy()
           predicted1 = predicted.cpu().numpy()
           for i in range(64):
             if flag==1:
              if target1[i]==predicted1[i]:
                 missed.append(i)
                 new_tensor = data[i].unsqueeze(0)
                 empty_tensor = torch.cat((empty_tensor, new_tensor), dim=0)
                 pred.append(predicted1[i])
                 targ.append(target1[i])
             else:
              if target1[i]!=predicted1[i]:
                 missed.append(i)
                 new_tensor = data[i].unsqueeze(0)
                 empty_tensor = torch.cat((empty_tensor, new_tensor), dim=0)
                 pred.append(predicted1[i])
                 targ.append(target1[i])
           break

  plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
  for i in range(0,10):
   plt.subplot(5, 2, i+1)
   frame1 = plt.gca()
   frame1.axes.xaxis.set_ticklabels([])
   frame1.axes.yaxis.set_ticklabels([])
   plt.imshow(np.transpose(((data[missed[i]].cpu()/2)+0.5).numpy(),(1,2,0)))
   plt.ylabel("GT:"+str(classes[target1[missed[i]]])+'\nPred:'+str(classes[predicted1[missed[i]]]))
  return empty_tensor, pred, targ
