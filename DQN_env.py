import namedtuple
import torch
import torchvision
from torchvision import datasets, transforms
from mnist_model import Net

MNIST_PIC_DIM = 28
SUCCESS_R = 100
FAIL_R = -1

modes = ['train', 'eval', 'test']

class Environment(object):
    #TODO add support to cuda if wanted
    def __init__(self, cnn_path = 'trained_models\mnist_cnn_epoch62.pt', seed = 0, block_size = 1, epsilon = 0.25, use_cuda = False, mode = 'train'):

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
        
        transform=transforms.Compose([
            transforms.ToTensor(),
            ])
        
        dataset1 = datasets.MNIST('../data', train = True, download = True,
            transform=transform)
        train_set, dev_set = torch.utils.data.random_split(dataset1, [50_000, 10_000])
        dataset2 = datasets.MNIST('../data', train = False, download = True,
            transform = transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        self.dev_loader = torch.utils.data.DataLoader(dev_set, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        assert mode in modes
        self.mode = mode

        self.train_iterator = iter(self.train_loader)
        self.dev_iterator = iter(self.dev_loader)
        self.test_iterator = iter(self.test_loader)
    
        self.cur_obs = namedtuple('image', 'label')

        self.width = MNIST_PIC_DIM
        self.height = MNIST_PIC_DIM
        self.block_size = block_size
        self.eps = epsilon

        self.cnn = Net().to(device)
        self.cnn.load_state_dict(torch.load(cnn_path))
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

    def get_screen(self):
        return return self.cur_obs.image

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

        self.cur_obs = data[0], target[0]#maybe this 0 will do problems but if removed probably need to be changes in other references to the cur_obs

        return self.cur_obs.image

    def act(self, action: int):
        #TODO adapt to block_size != 1
        sign = 1 if action % 2 == 0 else -1
        pixel = action // 2
        x, y = pixel // self.height, pixel % self.width
        org_label = self.cur_obs.label
        self.cur_obs.image[x][y] = torch.clamp(self.cur_obs.image[x][y] + sign * self.eps, 0, 1) #will it work, will it not, a mistery
        self.cur_obs.label = torch.argmax(self.cnn(self.cur_obs.image))

        reward = SUCCESS_R if self.cur_obs.label != org_label else FAIL_R

        return cur_obs.image, reward, self.cur_obs.label == org_label

    def set_mode(self, mode):
        assert mode in modes
        self.mode = mode

    @property
    def action_space(self):
        return 2 * self.width * self.height // (self.block_size ** 2)
