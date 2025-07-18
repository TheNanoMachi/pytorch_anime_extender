import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage
import os


latent_vector_size = 128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_size, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
 
    def forward(self, input):
        output = self.main(input)
        return output
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
        )
 
    def forward(self, input):
        output = self.main(input)
        return output

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

D = Discriminator().to(device)

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

G = Generator().to(device)

learning_rate = 0.0001
G_optimizer = optim.Adam(G.parameters(), lr = learning_rate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr = learning_rate, betas=(0.5, 0.999))
scheduler_G = lr_scheduler.StepLR(G_optimizer, step_size=10, gamma=0.1)
scheduler_D = lr_scheduler.StepLR(D_optimizer, step_size=10, gamma=0.1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias) 

D.apply(weights_init)
G.apply(weights_init)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
LOSS_G = []
LOSS_D = []
epochs = 1
epsilon = 100

criterion=nn.BCELoss()

# Training loop
train = False
if train:
    for epoch in tqdm(range(epochs)):
        print(epoch)
        for real_data in dataloader:
            real_data = real_data.to(device)
            noise =torch.randn(batch_size, latent_vector_size, 1, 1, device=device)
            fake_data = G(noise)
            
            # Discriminator predictions for real and fake data
            real_predictions = D(real_data)
            fake_predictions = D(fake_data)
    
            # Discriminator loss for real and fake data
            loss_D_real = criterion(real_predictions, torch.ones(len(real_predictions), 1).to(device))
            loss_D_fake = criterion(fake_predictions, torch.zeros(len(fake_predictions), 1).to(device))
            
            # Overall discriminator loss
            loss_D = (loss_D_fake + loss_D_real) / 2
            LOSS_D.append(loss_D.detach().item())
            
            # Backpropagation and optimizer update for discriminator
            D.zero_grad()
            loss_D.backward(retain_graph=True)
            D_optimizer.step()
            
            # Training the generator
            output = D(fake_data)
            loss_G = criterion(output, torch.ones(len(output), 1).to(device))
            LOSS_G.append(loss_G.detach().item())
        
            # Backpropagation and optimizer update for generator
            G.zero_grad()
            loss_G.backward()
            G_optimizer.step()
        
        # Using LR Scheduler
        scheduler_G.step()
        scheduler_D.step()
        
        # Displaying Images
        Xhat = G(noise).to(device).detach()
        plot_image_batch(Xhat)
        print("Epoch:", epoch)
        print(get_accuracy(real_data, Xhat))
        
        # Saving the model
        torch.save(D.state_dict(), 'D.pth')
        torch.save(G.state_dict(), 'G.pth')
else:
    D = Discriminator()
    D.load_state_dict(torch.load("D_trained.pth", map_location=torch.device('cpu')))
    G = Generator()
    G.load_state_dict(torch.load("G_trained.pth", map_location=torch.device('cpu')))

z = torch.randn(batch_size, latent_vector_size, 1, 1)

Xhat = G(z).detach()
plot_image_batch(Xhat)