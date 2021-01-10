from collections import namedtuple
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from mnist_model import Net

MNIST_PIC_DIM = 28
SUCCESS_R = 10_000
FINISH_R = 0
STEP_R = 0

modes = ['train', 'eval', 'test']

Observation = namedtuple('Observation', ['image', 'cur_label', 'org_label', 'actions'])


class Environment(object):
    def __init__(self, cnn_path='trained_models\mnist_cnn_epoch62.pt', seed=0, block_size=1, epsilon=0.25,
                 use_cuda=False, mode='train'):

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        train_kwargs = {'batch_size': 1}
        test_kwargs = {'batch_size': 1}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset1 = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)
        train_set, dev_set = torch.utils.data.random_split(dataset1, [50_000, 10_000])
        dataset2 = datasets.MNIST('../data', train=False, download=True,
                                  transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        self.dev_loader = torch.utils.data.DataLoader(dev_set, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        assert mode in modes
        self.mode = mode

        self.train_iterator = iter(self.train_loader)
        self.dev_iterator = iter(self.dev_loader)
        self.test_iterator = iter(self.test_loader)

        self.cur_obs = Observation(None, None, None, None)

        self.width = MNIST_PIC_DIM
        self.height = MNIST_PIC_DIM
        self.block_size = block_size
        self.eps = epsilon

        device = torch.device("cuda" if use_cuda else "cpu")

        self.cnn = Net().to(device)
        self.cnn.load_state_dict(torch.load(cnn_path))
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

    def get_screen(self):
        return self.cur_obs.image, self.cur_obs.actions

    def step(self, action: int):
        observation, reward, done = self.act(action)
        return observation, reward, done

    def reset(self):
        """
        :return: observation array
        """
        data, target = None, None
        try:
            if self.mode == 'train':
                data, target = next(self.train_iterator)
            elif self.mode == 'eval':
                data, target = next(self.dev_iterator)
            else:
                data, target = next(self.test_iterator)

        except StopIteration:
            if self.mode == 'train':
                self.train_iterator = iter(self.train_loader)
                data, target = next(self.train_iterator)
            elif self.mode == 'eval':
                self.dev_iterator = iter(self.dev_loader)
                data, target = next(self.dev_iterator)
            else:
                self.test_iterator = iter(self.test_loader)
                data, target = next(self.test_iterator)

        target = target.item()

        if torch.argmax(self.cnn(data.cuda())) != target:
            return self.reset()

        self.cur_obs = Observation(data.numpy(), target, target, np.zeros(self.action_space))

        return (self.cur_obs.image, self.cur_obs.actions)

    def act(self, action: int):
        if action == self.action_space - 1:  # reward 0 might cause choosing only the terminating action
            self.cur_obs.actions[-1] = 1
            return self.cur_obs.image, FINISH_R, True

        repeated_action = False
        if action != self.action_space - 1:
            if self.cur_obs.actions[2 * (action // 2)] == 1 or self.cur_obs.actions[2 * (action // 2) + 1] == 1:
                repeated_action = True
            self.cur_obs.actions[2 * (action // 2)] = self.cur_obs.actions[2 * (action // 2) + 1] = 1

        correct_label = self.cur_obs.cur_label == self.cur_obs.org_label
        sign = 1 if action % 2 == 0 else -1

        if not repeated_action:
            pixel = action // 2
            x, y = pixel // (self.height // self.block_size), pixel % (self.width // self.block_size)
            for i in range(self.block_size):
                for j in range(self.block_size):
                    self.cur_obs.image[0, 0][x + i][y + j] = np.clip(
                        self.cur_obs.image[0, 0][x + i][y + j] + sign * self.eps, 0, 1)

        pred = torch.exp(self.cnn(torch.FloatTensor(self.cur_obs.image).cuda())[0])
        new_label = torch.argmax(pred).item()
        new_label = new_label if correct_label else self.cur_obs.cur_label

        self.cur_obs = Observation(self.cur_obs.image, new_label, self.cur_obs.org_label, self.cur_obs.actions)

        mask = np.ones(10)
        mask[self.cur_obs.org_label] = -1
        reward = np.dot(mask, pred.cpu().numpy())
        reward = SUCCESS_R if self.cur_obs.cur_label != self.cur_obs.org_label and correct_label else reward
        reward = FINISH_R if repeated_action and self.mode == "eval" else reward

        return (self.cur_obs.image, self.cur_obs.actions), reward, False if self.mode == "train" else repeated_action  # repeated_action

    def set_mode(self, mode):
        assert mode in modes
        self.mode = mode

    @property
    def action_space(self):
        return 2 * self.width * self.height // (self.block_size ** 2) + 1
