import torch 
import torch.nn.functional as F

from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dp1 = nn.Dropout2d(0.25)
        self.dp2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, 2)

        x = self.dp1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp2(x)
        x = self.fc2(x)
        return x 