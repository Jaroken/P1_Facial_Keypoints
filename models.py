## TODO: define the convolutional neural network architecture

import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
    



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        imagesize = 35
        channelinput = 1
        outputchannel = 28
        kernel = 3
        stride = 1
        
        
        self.conv1 = nn.Conv2d(channelinput, outputchannel, kernel)
        dim = math.floor((imagesize - kernel)/stride)+1
        #print(dim)
        
        self.norm1 = nn.BatchNorm2d(28)
        
        pool_stride = 2
        pool_kernel = 2
        self.pool1 = nn.MaxPool2d(pool_kernel, pool_stride)
        dim = math.floor(dim/pool_stride)
        #print(dim)
        
        kernel = 3
        channelinput = outputchannel
        outputchannel = 28*2
        self.conv2 = nn.Conv2d(channelinput, outputchannel, kernel)
        dim = math.floor((dim - kernel)/stride)+1
        #print(dim)
        
        self.norm2 = nn.BatchNorm2d(28*2)
        
        pool_stride = 2
        pool_kernel = 2
        self.pool2 = nn.MaxPool2d(pool_kernel, pool_stride)
        dim=math.floor(dim/pool_stride)
        #print(dim)
        
        kernel = 3
        channelinput = outputchannel
        outputchannel = 28*3
        self.conv3 = nn.Conv2d(channelinput, outputchannel, kernel)
        dim = math.floor((dim - kernel)/stride)+1
        #print(dim)
        
        self.norm3 = nn.BatchNorm2d(28*3)
        
        pool_stride = 2
        pool_kernel = 2
        self.pool3 = nn.MaxPool2d(pool_kernel, pool_stride)
        dim=max([math.floor(dim/pool_stride),1])
        #print(dim)
        
        self.norm4 = nn.BatchNorm2d(28*4)
        
        kernel = 5
        channelinput = outputchannel
        outputchannel = 28*4
        self.conv4 = nn.Conv2d(channelinput, outputchannel, kernel)
        dim = math.floor((dim - kernel)/stride)+1
        #print(dim)
        
        pool_stride = 2
        pool_kernel = 2
        self.pool4 = nn.MaxPool2d(pool_kernel, pool_stride)
        dim=max([math.floor(dim/pool_stride),1])
        #print(dim)
        
        kernel = 5
        channelinput = outputchannel
        outputchannel = 28*5
        self.conv5 = nn.Conv2d(channelinput, outputchannel, kernel)
        dim = math.floor((dim - kernel)/stride)+1
        
        pool_stride = 2
        pool_kernel = 2
        self.pool5 = nn.MaxPool2d(pool_kernel, pool_stride)
        dim=max([math.floor(dim/pool_stride),1])
        
        # dropout with 20%
        self.drop = nn.Dropout(p=0.3)
        
        
        # fully-connected layer
 
        self.fc1 = nn.Linear(1260, int(round(1260/2, 0)))
        self.fc2 = nn.Linear(int(round(1260/2, 0)), int(round(1260/3, 0)))
        self.fc3 = nn.Linear(int(round(1260/3, 0)), 136)

    # define the feedforward behavior
    def forward(self, x):
        # one conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.norm1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = self.norm2(x)
        #print(x.shape)
        x = self.norm3(self.pool3(F.relu(self.conv3(x))))
        
        x = self.norm4(self.pool4(F.relu(self.conv4(x))))
        
        x = self.pool5(F.relu(self.conv5(x)))
        
        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        #print(x.shape)
        # dropout
       
        x = self.drop(F.relu(self.fc1(x)))
        
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        # final output
        return x

