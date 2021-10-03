
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

class ffn(nn.Module):
   
    def __init__(self):
        super(ffn, self).__init__()
        self.fc1 = nn.Linear(512*3, 12)
        self.fc2 = nn.Linear(12, 2)
        self.LeakyReLU = nn.LeakyReLU()
       
       
    def forward(self, x):
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        return x
