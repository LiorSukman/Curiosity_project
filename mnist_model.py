from __future__ import print_function
import argparse
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import numpy as np

PATIENCE = 20
PATH = 'trained_models\\'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2704, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, verbos=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if verbos:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action = 'store_true', default = True,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For Loading the Model Instead of Training')
    parser.add_argument('--load-path', type=str, default="mnist_cnn.pt",
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

    # create model, initialize optimizer
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    dev_losses = []
    train_losses = []
    dev_accuracy = []
    train_accuracy = []

    if not args.load_model:  # don't need to load
        best_epoch = 0
        best_loss = float('inf')
        start_time = time.time()
        # run training
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            train_loss, train_acc = test(model, device, train_loader)
            dev_loss, dev_acc = test(model, device, dev_loader)
            dev_losses.append(dev_loss)
            dev_accuracy.append(dev_acc)
            train_losses.append(train_loss)
            train_accuracy.append(train_acc)
            if dev_loss < best_loss:  # found better epoch
                best_loss = dev_loss
                best_epoch = epoch
            if args.save_model:  # need to save model
                model_name = 'mnist_cnn_epoch%d.pt' % (epoch)
                torch.save(model.state_dict(), PATH + model_name)
            if best_epoch + PATIENCE <= epoch:  # no improvment in the last PATIENCE epochs
                print('No improvement was done in the last %d epochs, breaking...' % PATIENCE)
                break
        end_time = time.time()
        print('Training took %.3f seconds' % (end_time - start_time))
        print('Best model was achieved on epoch %d' % best_epoch)
        model_name = 'mnist_cnn_epoch%d.pt' % (best_epoch)
        model.load_state_dict(torch.load(PATH + model_name))  # load model from best epoch

        epochs = np.arange(1, len(dev_losses) + 1)
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('Loss', color = color)
        ax1.plot(epochs, dev_losses, 'b-', label = 'dev loss')
        ax1.plot(epochs, train_losses, 'b--', label = 'train loss')
        ax1.tick_params(axis='y', labelcolor = color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('Accuracy', color = color)  # we already handled the x-label with ax1
        ax2.plot(epochs, dev_accuracy, 'r-', label='dev accuracy')
        ax2.plot(epochs, train_accuracy, 'r--', label='train accuracy')
        ax2.tick_params(axis='y', labelcolor = color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.legend()
        ax2.legend()
        plt.title('Classifying CNN training - Loss and Accuracy vs. Epoch')
        plt.show()
    else:  # need to load
        model.load_state_dict(torch.load(PATH + args.load_path))

    print('Testing test set...')
    test(model, device, test_loader)


if __name__ == '__main__':
    main()
