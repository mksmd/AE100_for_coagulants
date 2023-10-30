import torch
from torch import nn
from torch.nn import functional as F


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        # Tensor - (N_batch, Channels = 1, Height = 150, Width = 21)
        self.conv0 = nn.Conv2d(1, 21, kernel_size=(1, 21), stride=(1, 1))
        self.conv0_bn = nn.BatchNorm2d(21)

        # Tensor - (N_batch, Channels = 21, Height = 150, Width = 1)
        self.conv1 = nn.Conv2d(21, 150, kernel_size=(2, 1), stride=(2, 1))
        self.conv1_bn = nn.BatchNorm2d(150)

        # Tensor - (N_batch, Channels = 150, Height = 75, Width = 1)
        self.conv2 = nn.Conv2d(150, 500, kernel_size=(5, 1), stride=(5, 1))
        self.conv2_bn = nn.BatchNorm2d(500)

        # Tensor - (N_batch, Channels = 500, Height = 15, Width = 1)
        self.conv3 = nn.Conv2d(500, 1000, kernel_size=(15, 1), stride=(1, 1))
        self.conv3_bn = nn.BatchNorm2d(1000)
        # Tensor - (N_batch, Channels = 1000, Height = 1, Width = 1)

        self.efc0 = nn.Linear(1000, 100)
        self.efc0_bn = nn.BatchNorm1d(100)

        ###

        self.dfc0 = nn.Linear(100, 1000)
        self.dfc0_bn = nn.BatchNorm1d(1000)

        # Tensor - (N_batch, Channels = 1000, Height = 1, Width = 1)
        self.deconv0 = nn.ConvTranspose2d(1000, 500, kernel_size=(15, 1), stride=(1, 1))
        self.deconv0_bn = nn.BatchNorm2d(500)

        # Tensor - (N_batch, Channels = 500, Height = 15, Width = 1)
        self.deconv1 = nn.ConvTranspose2d(500, 150, kernel_size=(5, 1), stride=(5, 1))
        self.deconv1_bn = nn.BatchNorm2d(150)

        # Tensor - (N_batch, Channels = 150, Height = 75, Width = 1)
        self.deconv2 = nn.ConvTranspose2d(150, 21, kernel_size=(2, 1), stride=(2, 1))
        self.deconv2_bn = nn.BatchNorm2d(21)

        # Tensor - (N_batch, Channels = 21, Height = 150, Width = 1)
        self.deconv3 = nn.ConvTranspose2d(21, 1, kernel_size=(1, 21), stride=(1, 1))
        self.deconv3_bn = nn.BatchNorm2d(1)
        # Tensor - (N_batch, Channels = 1, Height = 50, Width = 21)

    def encode(self, x):
        h0 = F.relu(self.conv0_bn(self.conv0(x.view(-1, 1, 150, 21))))
        h1 = F.relu(self.conv1_bn(self.conv1(h0)))
        h2 = F.relu(self.conv2_bn(self.conv2(h1)))
        h3 = F.relu(self.conv3_bn(self.conv3(h2)))
        return torch.sigmoid(self.efc0_bn(self.efc0(h3.view(-1, 1000))))

    def decode(self, x):
        h0 = F.relu(self.dfc0_bn(self.dfc0(x)))
        h1 = F.relu(self.deconv0_bn(self.deconv0(h0.view(-1, 1000, 1, 1))))
        h2 = F.relu(self.deconv1_bn(self.deconv1(h1)))
        h3 = F.relu(self.deconv2_bn(self.deconv2(h2)))
        return F.softmax(self.deconv3_bn(self.deconv3(h3)), dim=3).view(-1, 150, 21)

    def forward(self, x):
        h = self.encode(x)
        x = self.decode(h)
        return x
