"""
The contents of this file are duplicated in the attached notebook for readability
For most tests, comparisons and visualization we advise you to refer to it.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np
from mnist_model import Net

PATIENCE = 20
PATH = 'trained_models\\'
PIC_DIM = 28
EPS = 0.25

# regularization variables
C = 1
GAMMA = 0.0
THR = 0.0


def custom_loss(output, noise, target, cnn):
    preds = cnn(output)
    loss = C / F.nll_loss(preds, target) + GAMMA * torch.sum(torch.abs(noise)) / len(noise)
    return loss


class PGEN_NN(nn.Module):
    def __init__(self):
        super(PGEN_NN, self).__init__()
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
        x = torch.flatten(x, 1)  # 2704
        x = self.fc1(x)  # 1024
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # 784
        x = torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM))
        output = torch.clamp(torch.tanh(x) * EPS + org, 0, 1)
        return output, x

    def generate(self, x, device):
        """
        this function is the same as forward except it uses sign instead of tanh for noise generation
        for discussion regarding the reasons please refer to the submitted pdf document
        threshold value THR is also used to limit the noise added (please refer to appendix C in the pdf document)
        """
        org = x
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, PIC_DIM, PIC_DIM))
        x = torch.tanh(x).cpu().numpy()
        x = torch.from_numpy(np.where(np.abs(x) > THR, x, 0)).to(device)
        output = torch.clamp(torch.sign(x) * EPS + org, 0, 1)
        return output, x


def train(args, model, device, train_loader, optimizer, epoch, cnn, verbos=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, noise = model(data)
        loss = custom_loss(output, noise, target, cnn)  # F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if verbos and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, cnn, verbos=True):
    model.eval()
    # values for statistics
    test_loss = 0
    noise_count = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, noise = model.generate(data, device)

            # update statistics values
            noise_count += torch.sum(torch.abs(noise))
            test_loss += custom_loss(output, noise, target,
                                     cnn) * len(data)
            output = cnn(output)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbos:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), noise average = {:.4f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            noise_count / len(test_loader.dataset)))

    return test_loss, correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Perturbation Network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For Loading the Model Instead of Training')
    parser.add_argument('--load-path', type=str, default="",
                        help='For Loading the Model Instead of Training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")  # device to run the model on

    # organize parsed data
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        torch.cuda.manual_seed(args.seed)

    # get datasets and create loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    train_set, dev_set = torch.utils.data.random_split(dataset1, [50_000, 10_000],
                                                       generator=torch.Generator().manual_seed(42))
    dataset2 = datasets.MNIST('../data', train=False, download=True,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    dev_loader = torch.utils.data.DataLoader(dev_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    cnn = Net().to(device)
    cnn.load_state_dict(torch.load('trained_models\mnist_cnn_epoch62.pt'))
    for param in cnn.parameters():
        param.requires_grad = False
    cnn.eval()

    # create model, initialize optimizer
    model = PGEN_NN().to(device)
    optimizer = optim.Adadelta(model.parameters())

    if not args.load_model:  # don't need to load
        best_epoch = 0
        best_loss = float('inf')
        start_time = time.time()
        # run training
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, cnn)
            dev_loss, _ = test(model, device, dev_loader, cnn)
            if dev_loss < best_loss:  # found better epoch
                best_loss = dev_loss
                best_epoch = epoch
            if args.save_model:  # need to save model
                model_name = 'pgen_nn_epoch%d.pt' % epoch
                torch.save(model.state_dict(), PATH + model_name)

            """
            # add this section if want to use patience 
            if best_epoch + PATIENCE <= epoch:  # no improvement in the last PATIENCE epochs
                print('No improvement was done in the last %d epochs, breaking...' % PATIENCE)
                break
            """

        end_time = time.time()
        print('Training took %.3f seconds' % (end_time - start_time))
        print('Best model was achieved on epoch %d' % best_epoch)
        model_name = 'pgen_nn_epoch%d.pt' % best_epoch
        model.load_state_dict(torch.load(PATH + model_name))  # load model from best epoch
    else:  # need to load
        model.load_state_dict(torch.load(PATH + args.load_path))

    print('Testing test set...')
    test(model, device, test_loader, cnn)


if __name__ == '__main__':
    main()
