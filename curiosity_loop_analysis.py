import numpy as np
import argparse
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt

from dataset import Dataset
from mnist_model import Net
from NN_pert_gen import PGEN_NN, PATH, train, test

use_cuda = False  # torch.cuda.is_available() with small datasets as used here better without because of overhead


def create_graph(errors):
    y = [np.mean(error) for error in errors]

    w = 20
    y = np.convolve(y, np.ones(w), 'valid') / w
    x = np.arange(1, len(y) + 1)

    plt.plot(x, y)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Curiosity Loop')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=True,
                        help='For Loading the Model Instead of Training')
    parser.add_argument('--load-path', type=str, default=PATH,
                        help='For Loading the Model Instead of Training')
    parser.add_argument('--cnn-path', type=str, default="trained_models\\mnist_cnn_epoch62.pt",
                        help='For Loading the Classifying CNN')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")  # device to run the model on

    # organize parsed data
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # get datasets and create loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, download=True,
                              transform=transform)

    train_images = np.array([dataset1[i][0].numpy() for i in range(len(dataset1))])
    train_labels = np.array([dataset1[i][1] for i in range(len(dataset1))])
    test_images = np.array([dataset2[i][0].numpy() for i in range(len(dataset2))])
    test_labels = np.array([dataset2[i][1] for i in range(len(dataset2))])

    cnn = Net().to(device)
    cnn.load_state_dict(torch.load(args.cnn_path))
    for param in cnn.parameters():
        param.requires_grad = False
    cnn.eval()

    Q = np.load(args.load_path + 'policy.npy')
    errors = list(np.load(args.load_path + 'errors.npy', allow_pickle=True))
    losses = list(np.load(args.load_path + 'losses.npy', allow_pickle=True))

    print(Q*100)

    create_graph(losses)


if __name__ == '__main__':
    main()
