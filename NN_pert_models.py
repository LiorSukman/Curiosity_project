from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


PIC_DIM = 28
EPS = 0.25


class PGEN_NN1(nn.Module):
    def __init__(self):
        super(PGEN_NN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2704, 1024)
        self.fc2 = nn.Linear(1024, 784)

    def forward(self, x):
        org = x
        x = self.conv1(x)  # 16,26,26
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 16,13,13
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM))
        output = torch.clamp(torch.tanh(x) * EPS + org, 0, 1)
        return output

    def generate(self, x):
        org = x
        x = self.conv1(x)  # 16,26,26
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 16,13,13
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM))
        output = torch.clamp(torch.sign(torch.tanh(x)) * EPS + org, 0, 1)
        return output

class PGEN_NN2(nn.Module):
    def __init__(self):
        super(PGEN_NN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(676, 338)
        self.fc2 = nn.Linear(338, 784)

    def forward(self, x):
        org = x
        x = self.conv1(x)  # 4,26,26
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 4,13,13
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM))
        output = torch.clamp(torch.tanh(x) * EPS + org, 0, 1)
        return output

    def generate(self, x):
        org = x
        x = self.conv1(x)  # 16,26,26
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 16,13,13
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM))
        output = torch.clamp(torch.sign(torch.tanh(x)) * EPS + org, 0, 1)
        return output

class PGEN_NN3(nn.Module):
    def __init__(self):
        super(PGEN_NN3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2704, 1024)
        self.fc2 = nn.Linear(1024, 784)

    def forward(self, x):
        org = x
        x = self.conv1(x)  # 32,26,26
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)  # 4,13,13
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM))
        output = torch.clamp(torch.tanh(x) * EPS + org, 0, 1)
        return output

    def generate(self, x):
        org = x
        x = self.conv1(x)  # 32,26,26
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)  # 4,13,13
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM))
        output = torch.clamp(torch.sign(torch.tanh(x)) * EPS + org, 0, 1)
        return output