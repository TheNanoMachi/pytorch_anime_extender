import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from  torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import lr_scheduler
from tqdm import tqdm
from IPython import display
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage
import os
from os import listdir
from pathlib import Path
import imghdr
import skillsnetwork

class Discriminator(nn.Module):
    def __init__(self,input_dim=1):
        super(Discriminator,self).__init__()
        self.l1=nn.Linear(1,input_dim)
    
    def forward(self, x):
        return torch.sigmoid(self.l1(x))

D=Discriminator() 

class Generator(nn.Module):
    def __init__(self,input_dim=1):
        super(Generator,self).__init__()
        self.l1=nn.Linear(1,input_dim)
    
    def forward(self, x):
        return self.l1(x)
G=Generator()

def plot_distribution(real_data,generated_data,discriminator=None,density=True):
    
    plt.hist(real_data.numpy(), 100, density=density, facecolor='g', alpha=0.75, label='real data')
    plt.hist(generated_data.numpy(), 100, density=density, facecolor='r', alpha=0.75,label='generated data q(z) ')
    
    if discriminator:
        max_=torch.max(real_data.max(),generated_data.max().detach())
        min_=torch.min(real_data.min(),generated_data.min().detach())
        x=torch.linspace(start=min_, end=max_, steps=100)
        plt.plot(x.numpy(),discriminator(x.view(-1,1)).detach().view(-1).numpy(),label='discriminator',color='k')
        plt.plot(x.numpy(),0.5*np.ones(x.shape),label='0.5',color='b')
        plt.legend()
        plt.show()

device = torch.device("cpu")
def plot_image_batch(my_batch):

  fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
  img_num=0
  for i in range(8):
      for j in range(8):
          ax = axes[i][j]
          img_num+=1
  
          ax.imshow(np.transpose(vutils.make_grid(my_batch[img_num].to(device), padding=2, normalize=True).cpu(),(1,2,0)))
  plt.show()

def get_accuracy(X,Xhat):
    total=0
    py_x=D(X)
    total=py_x.mean()
    py_x=D(Xhat)
    total+=py_x.mean()
    return total/2

current_directory = os.getcwd()
directory=os.path.join(current_directory ,'images')
[filename for filename in os.listdir(directory) if filename.endswith('.jpg') ]

class Dataset(Dataset):
    def __init__(self, transform=None):
      current_directory = os.getcwd()
      directory=os.path.join(current_directory ,'images')

      self.file_paths = [os.path.join(directory,filename) for filename in os.listdir(directory) if filename.endswith('.jpg') ]
      self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image_path = self.file_paths[index]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image
dataset=Dataset()

image_size = 64
transform=transforms.Compose([
                               transforms.Resize((64, 64)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

dataset=Dataset(transform=transform)

batch_size = 256
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)

real_batch = next(iter(dataloader))
real_batch.shape

plot_image_batch(real_batch)