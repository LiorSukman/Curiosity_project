import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

class DQN(nn.Module):
    #TODO check hyperparameters
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2704, 1024)
        self.fc2 = nn.Linear(1024 + self.n_action, self.n_action)

        """
        self.conv1 = nn.Conv2d(1, 8, kernel_size = 7, stride = 1, padding = 0)  # (In Channel, Out Channel, ...) #maybe to large of a kernel and stride
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 4, stride = 1, padding = 0)
        
        self.affine1 = nn.Linear(5776, 2048) #might need to change input size 
        self.affine2 = nn.Linear(2048 + self.n_action, self.n_action)
        """

    def forward(self, x):
        images, actions = x
        """
        h = F.relu(self.conv1(images))
        h = F.relu(self.conv2(h))

        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        h = torch.cat((h, actions), dim = 1)
        h = self.affine2(h)
        """
        h = self.conv1(images)
        h = F.relu(h)
        h = F.max_pool2d(h, 2)
        #x = self.dropout1(x)
        h = torch.flatten(h, 1)
        h = self.fc1(h)
        h = F.relu(h)
        #x = self.dropout2(x)
        h = torch.cat((h, actions), dim = 1)
        h = self.fc2(h)
        return h


class LSTMDQN(nn.Module):
    #TODO check hyperparameters
    #TODO add support to cuda if wanted
    def __init__(self, n_action):
        super(LSTMDQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(1, 32, kernel_size = 8, stride = 1, padding = 1)  # (In Channel, Out Channel, ...) #maybe to large of a kernel and stride
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)

        self.lstm = nn.LSTM(16, LSTM_MEMORY, 1)  # (Input, Hidden, Num Layers) #might need to change input size

        self.affine1 = nn.Linear(LSTM_MEMORY * 64, 512)
        self.affine2 = nn.Linear(512, self.n_action) #maybe something larger than 512 as we have a lot of actions

    def forward(self, x, hidden_state, cell_state):
        # CNN
        h = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        h = F.relu(F.max_pool2d(self.conv2(h), kernel_size = 2, stride = 2))
        h = F.relu(F.max_pool2d(self.conv3(h), kernel_size = 2, stride = 2))
        h = F.relu(F.max_pool2d(self.conv4(h), kernel_size = 2, stride = 2))

        # LSTM
        h = h.view(h.size(0), h.size(1), 16)  # (32, 64, 4, 4) -> (32, 64, 16)
        h, (next_hidden_state, next_cell_state) = self.lstm(h, (hidden_state, cell_state))
        h = h.view(h.size(0), -1)  # (32, 64, 256) -> (32, 16348)

        # Fully Connected Layers
        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        # h = F.relu(self.affine2(h.view(h.size(0), -1)))
        h = self.affine2(h)
        return h, next_hidden_state, next_cell_state

    def init_states(self) -> [Variable, Variable]:
        hidden_state = Variable(torch.zeros(1, 64, LSTM_MEMORY))#.cuda()) 
        cell_state = Variable(torch.zeros(1, 64, LSTM_MEMORY))#.cuda())
        return hidden_state, cell_state

    def reset_states(self, hidden_state, cell_state):
        hidden_state[:, :, :] = 0
        cell_state[:, :, :] = 0
        return hidden_state.detach(), cell_state.detach()

    
