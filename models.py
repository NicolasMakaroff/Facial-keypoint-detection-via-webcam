## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        ## Shape of the convolutional layer
        # K - out_channel
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding
        # W - width/height
        
        # initially : 224*224 pixels image
        # output size = (W-F+2P)/S + 1
        
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)       ## output size = (224-3+2)/1 +1 = 224
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)      ## output size = (112-3+2)/1 +1 = 112
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)     ## output size = (56-3+2)/1 +1 = 56
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)    ## output size = (28-3+2)/1 +1 = 28
        
        
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(256*14*14,4048)
        self.bn1 = nn.BatchNorm1d(4048)
        self.fc2 = nn.Linear(4048,1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024,512) 
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512,136)
    
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0),-1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        
        return x
