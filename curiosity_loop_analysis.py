import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
from scipy.stats import sem

from mnist_model import Net
from NN_pert_gen import PATH
from curiosity_loop import curiosity_loop_test

use_cuda = torch.cuda.is_available()


def compare_q_graph(reps=10):
    """
        Create a graph of comparison between the random policy and learned policy

        Parameters
        ----------
        reps : int, default=10
            the number of repetitions for the learned policy and random policy
    """
    steps = np.arange(1, reps + 1)

    # load relevant files
    q_errors = np.load(PATH + 'q_errors.npy', allow_pickle=True)
    r_errors = np.load(PATH + 'r_errors.npy', allow_pickle=True)

    plt.errorbar(steps, q_errors.mean(axis=0), yerr=sem(q_errors, axis=0), label='Policy based', c='k')
    plt.errorbar(steps, r_errors.mean(axis=0), yerr=sem(r_errors, axis=0), label='Random', c='b')
    plt.xlabel('Step')
    plt.ylabel('Error Fraction')
    plt.title('Error Fraction by Policy Step')
    plt.legend()
    plt.show()
    plt.clf()

    # uncomment the next section for losses comparison graph
    """
    q_losses = np.load(PATH + 'q_losses.npy', allow_pickle=True)
    r_losses = np.load(PATH + 'r_losses.npy', allow_pickle=True)

    q_losses = np.array([np.array([i.item() for i in j]) for j in q_losses])
    r_losses = np.array([np.array([i.item() for i in j]) for j in r_losses])

    plt.errorbar(steps, q_losses.mean(axis=0), yerr=sem(q_losses, axis=0), label='Policy based', c='k')
    plt.errorbar(steps, r_losses.mean(axis=0), yerr=sem(r_losses, axis=0), label='Random', c='b')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss by Step')
    plt.legend()
    plt.show()
    plt.clf()
    """


def compare_q_run(Q, train_images, train_labels, test_loader, args, device, cnn, reps=10, dev_loader=None):
    """
        Compare learned policy with random policy,
        results are saved under PATH as q_errors.npy, q_losses.npy, q_c_sels.npy
        which are the Q based errors, losses and selected actiond respectively
        and r_errors.npy, r_losses.npy, r_c_sels.npy which are the random policy
        based errors, losses and selected actiond respectively.

            Parameters
            ----------
            Q : numpy array
                two dimensional ndarray of the learned Q table
            train_images : numpy array
                images of the data
            train_labels : numpy array
                labels of the data corresponding to train_images
            test_loader : torch dataloader
                dataloader to use for evaluation
            args : parser
                parser containig hyperparameters
            device : torch device
                device the model is ran on, can be cpu or gpu
            cnn : torch model
                a CNN used for classifying the perturbed data, please see part 2 submission for elaboration
            reps : int, default=10
                the number of repetitions for the learned policy and random policy
            dev_loader : torch dataloader, default=None
                if given, used for model selection, otherwise train data is used

    """
    # initialize dataholders
    q_errors = []
    q_losses = []
    q_c_sels = []

    # start Q evaluation
    for i in range(reps):
        q_error, q_loss, q_c_sel = curiosity_loop_test(Q, train_images, train_labels, test_loader, args,
                                                       device, cnn, dev_loader=dev_loader)
        q_errors.append(q_error)
        q_losses.append(q_loss)
        q_c_sels.append(q_c_sel)

        print(f'finished repetition {i + 1} using Q')

    # save results
    np.save(PATH + 'q_errors', np.array(q_errors))
    np.save(PATH + 'q_losses', np.array(q_losses))
    np.save(PATH + 'q_c_sels', np.array(q_c_sels))

    # initialize dataholders
    r_errors = []
    r_losses = []
    r_c_sels = []

    # start random policy evaluation
    for i in range(reps):
        r_error, r_loss, r_c_sel = curiosity_loop_test(Q, train_images, train_labels, test_loader, args,
                                                       device, cnn, randomize=True, dev_loader=dev_loader)
        r_errors.append(r_error)
        r_losses.append(r_loss)
        r_c_sels.append(r_c_sel)

        print(f'finished repetition {i + 1} randomly')

    # save results
    np.save(PATH + 'r_errors', np.array(r_errors))
    np.save(PATH + 'r_losses', np.array(r_losses))
    np.save(PATH + 'r_c_sels', np.array(r_c_sels))


def create_graph(data, w=1, xlabel='', ylabel='', title=''):
    """
    Create a graph of the information

        Parameters
        ----------
        data : list of list of float
            the data to plot
        w : int
            window size
        xlabel : str
            string representing the label of the x axis
        ylabel : str
            string representing the label of the y axis
        title : str
            string representing the title

    """
    y = [np.mean(d) for d in data]  # average steps in episode

    y = np.convolve(y, np.ones(w), 'valid') / w
    x = np.arange(1, len(y) + 1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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
    train_set, dev_set = torch.utils.data.random_split(dataset1, [50_000, 10_000],
                                                       generator=torch.Generator().manual_seed(42))
    dataset2 = datasets.MNIST('../data', train=False, download=True,
                              transform=transform)

    train_images = np.array([train_set[i][0].numpy() for i in range(len(train_set))])
    train_labels = np.array([train_set[i][1] for i in range(len(train_set))])

    test_kwargs = {'batch_size': args.test_batch_size}
    dev_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    cnn = Net().to(device)
    cnn.load_state_dict(torch.load(args.cnn_path))
    for param in cnn.parameters():
        param.requires_grad = False
    cnn.eval()

    Q = np.load(args.load_path + 'policy.npy')
    errors = list(np.load(args.load_path + 'errors.npy', allow_pickle=True))
    losses = list(np.load(args.load_path + 'losses.npy', allow_pickle=True))

    print(Q)

    create_graph(errors, w=20, xlabel='Step', ylabel='Average Error Fraction', title='Model\'s Average Error Fraction by Curiosity Loop Episode')
    create_graph(losses, w=20, xlabel='Step', ylabel='Avergae Loss', title='Model\'s Average Loss by Curiosity Loop Step')

    #uncomment if you do not want to rerun
    compare_q_run(Q, train_images, train_labels, test_loader, args, device, cnn, dev_loader=dev_loader)

    compare_q_graph()


if __name__ == '__main__':
    main()
