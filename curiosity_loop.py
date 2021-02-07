import numpy as np
import argparse
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import random

from dataset import Dataset
from mnist_model import Net
from NN_pert_gen import PGEN_NN, PATH, train, test

use_cuda = False  # with small datasets as used here better without because of overhead


def sample_episode(episode_size, train_images, train_labels, classes=np.arange(10)):
    """
    samples data of required size from given data

        Parameters
        ----------
        episode_size : int
            number of examples to be sampled
        train_images : numpy array
            images of the data
        train_labels : numpy array
            labels of the data corresponding to train_images
        classes : list of int, default: np.arange(10)
            the possible classes, used for balanced sampling

        Returns
        -------
        images : numpy array
            sampled images
        labels : numpy array
            sampled labels corresponding to the images
    """
    sample_size = episode_size // len(classes)  # calculate number of samples from each class
    indices = [np.nonzero(train_labels == c)[0] for c in classes]  # indices for examples with specific label

    images = []
    labels = []
    for inds in indices:
        meta_inds = np.random.choice(inds.flatten(), size=sample_size, replace=False)
        images.append(train_images[meta_inds])
        labels.append(train_labels[meta_inds])

    # stack sampled images and corresponding labels
    images = np.vstack(tuple(images))
    labels = np.hstack(tuple(labels))

    return images, labels


def split_data(x, y, percentage, classes):
    """
    splits given data to train and dev sets according to given ratio

        Parameters
        ----------
        x : numpy array
            images of the data
        y : numpy array of images
            labels correxponding to the images in x
        percentage : float
            fraction of samples to be in the dev set
        classes : list of int
            the possible classes, used for balanced sampling

        Returns
        -------
        x_train : numpy array of images
            sampled images for the training set
        y_train : numpy array of int
            sampled labels for the training set corresponding to the x_train
        x_test : numpy array of images
            sampled images for the dev set
        y_test : numpy array of int
            sampled labels for the dev set corresponding to the x_test
    """
    percentage = 1 - percentage
    indices = [np.nonzero(y == c)[0] for c in classes]
    x_train, x_test, y_train, y_test = [], [], [], []

    # partition each class to train and dev
    for inds in indices:
        np.random.shuffle(inds)
        x_train.append(x[inds[:int(percentage * len(inds))]])
        x_test.append(x[inds[int(percentage * len(inds)):]])
        y_train.append(y[inds[:int(percentage * len(inds))]])
        y_test.append(y[inds[int(percentage * len(inds)):]])

    # stack sets of images and corresponding labels
    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)

    return x_train, x_test, y_train, y_test


def train_model(model, optimizer, args, dataloader, device, cnn, name, verbos=False, dev_loader=None):
    """
    trains learner

        Parameters
        ----------
        model : torch model
            model to train
        optimizer : torch optimizer
            optimizer to use for steps
        args : parser
            parser containig hyperparameters (specifically number of epochs)
        dataloader : torch dataloader
            dataloader to use for training
        device : torch device
            device the model is ran on, can be cpu or gpu
        cnn : torch model
            a CNN used for classifying the perturbed data, please see part 2 submission for elaboration
        name : str
            a string for the name underwhich the model will be saved
        verbos : bool, default=False
            whether to print messages during trining
        dev_loader : torch dataloader, default=None
            if given, used for evaluation and model selection, otherwise dataloader is used
    """
    best_epoch = 0
    best_loss = float('inf')
    model_name = name + '_best.pt'
    start_time = time.time()
    # run training
    for epoch in range(1, args.epochs + 1):
        if epoch % 10 == 0:
            print(f'        In epoch {epoch}')
        train(args, model, device, dataloader, optimizer, epoch, cnn, verbos=verbos)
        if dev_loader is None:
            loss, accuracy = test(model, device, dataloader, cnn, verbos=verbos)
        else:
            loss, accuracy = test(model, device, dev_loader, cnn, verbos=verbos)
        if loss < best_loss:  # found better epoch
            best_loss = loss
            best_epoch = epoch
            torch.save(model.state_dict(), PATH + model_name)

    end_time = time.time()
    print('    Training took %.3f seconds' % (end_time - start_time))
    print('    Best model was achieved on epoch %d' % best_epoch)
    model.load_state_dict(torch.load(PATH + model_name))  # load model from best epoch


def curiosity_loop_test(Q, train_images, train_labels, test_loader, args, device, cnn, classes=np.arange(10),
                        randomize=False, dev_loader=None):
    """
    tests the performance of a policy

        Parameters
        ----------
        Q : numpy array
            Q table
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
        classes : list of int, default: np.arange(10)
            the possible classes, used for balanced sampling
        randomize : bool, default: False
            if True use random policy, otherwise use Q
        dev_loader : torch dataloader, default=None
            if given, used for model selection, otherwise train data is used

        Returns
        -------
        errors : list of float
            list of the errors in each step
        losses : list of float
            list of the loss values in each step
        c_sel : list of int
            list of selected actions in each step
    """
    # dataholders
    errors = []
    losses = []
    c_sel = []

    st = 10  # s0
    c_available = list(classes)
    while len(c_available) != 0:  # there are available actions
        if randomize:  # random policy
            at = random.sample(c_available, 1)[0]
            print(f'    randomly chose action {at}')
        else:  # actual policy
            qa = Q[st].copy()
            qa[c_sel] = -1  # for avoiding chosen actions, might not be generic but suits our case
            at = qa.argmax()
            print(f'    rl chose action {at} with Q value of {qa.max()}')
        c_available.remove(at)
        c_sel.append(at)
        train_dataset = Dataset(train_images, train_labels, c_sel, args.batch_size, use_cuda)
        train_loader = train_dataset.get_dataloader()
        print(f'    dataset size is {len(train_dataset)}')

        model = PGEN_NN().to(device)
        optimizer = optim.Adadelta(model.parameters())

        name = f'pgen_nn_test_randomize_{randomize}_at_{at}'
        train_model(model, optimizer, args, train_loader, device, cnn, name, verbos=False, dev_loader=dev_loader)
        loss_val, et = test(model, device, test_loader, cnn, verbos=False)
        errors.append(et)
        losses.append(loss_val)
        print(f'    error of the learner was {et}')
        st = at
        print('--------------')

    return errors, losses, c_sel


def curiosity_loop(train_images, train_labels, args, device, cnn, classes=np.arange(10)):
    """
    tests the performance of a policy

        Parameters
        ----------
        train_images : numpy array
            images of the data
        train_labels : numpy array
            labels of the data corresponding to train_images
        args : parser
            parser containig hyperparameters
        device : torch device
            device the model is ran on, can be cpu or gpu
        cnn : torch model
            a CNN used for classifying the perturbed data, please see part 2 submission for elaboration
        classes : list of int, default: np.arange(10)
            the possible classes, used for balanced sampling

        Returns
        -------
        Q : numpy array
            learned Q table
        errors : list of list of float
            list of the errors in each step in each episode
        losses : list of list of float
            list of the losses in each step in each episode
    """
    N_episode = args.n_episode  # number of samples for each episode
    N_iter = 10 * train_images.shape[0] // N_episode  # number of iterations (or episodes)
    gamma = args.discount_factor
    threshold = args.threshold

    if args.load_model:  # continue stopped learning process
        # load Q and data holders
        Q = np.load(args.load_path + 'policy.npy')
        q_first = np.load(args.load_path + 'q_first.npy')
        errors = list(np.load(args.load_path + 'errors.npy'))
        losses = list(np.load(args.load_path + 'losses.npy'))
        cur_episode = len(losses)

        random.seed(args.seed + cur_episode)
        np.random.seed(args.seed + cur_episode)
        torch.manual_seed(args.seed + cur_episode)
    else:  # start new learning process
        # initialize Q and data holders
        Q = np.ones((len(classes) + 1, len(classes)))
        q_first = np.zeros((len(classes) + 1, len(classes)))
        errors = []
        losses = []
        cur_episode = 0

    for i in range(cur_episode, N_iter):
        start_time = time.time()
        print(f'************Starting episode {i + 1}************')
        # sample episode
        episode_images, episode_labels = sample_episode(N_episode, train_images, train_labels, classes=classes)
        train_episode_images, dev_episode_images, train_episode_labels, dev_episode_labels = \
            split_data(episode_images, episode_labels, 0.2, classes)  # partition to train and dev
        e0 = 0.985  # see submitted report for elaboration
        st = 10  # s0
        c_sel = []  # selected classes/actions
        c_available = list(classes)  # available classes/actions

        # set exploration rate and learning rate according to schedule
        eps = 0.9 if i < N_iter / 4 else (0.5 if (N_iter / 4 <= i < N_iter / 2) else
                                          (0.3 if (N_iter / 2 <= i < 3 * N_iter / 4) else 0.1))
        alpha = 0.09 if i < N_iter / 4 else (0.05 if (N_iter / 4 <= i < N_iter / 2) else
                                             (0.01 if (N_iter / 2 <= i < 3 * N_iter / 4) else 0.005))
        dev_dataset = Dataset(dev_episode_images, dev_episode_labels, None, args.test_batch_size, use_cuda)
        dev_loader = dev_dataset.get_dataloader()
        t = 0
        error = []
        loss = []
        while len(c_available) != 0:  # while still has available actions
            print(f'    Starting time step {t}')
            if eps > random.random():  # explore
                at = random.sample(c_available, 1)[0]
            else:  # exploit
                qa = Q[st].copy()
                qa[c_sel] = threshold  # to make sure we do not choose the same action twice
                if qa.max() <= threshold:
                    break
                at = qa.argmax()
            print(f'    rl chose action {at}')
            # update available and selected moves
            c_available.remove(at)
            c_sel.append(at)
            # create Dataset and dataloader
            train_dataset = Dataset(train_episode_images, train_episode_labels, c_sel, args.batch_size, use_cuda)
            train_loader = train_dataset.get_dataloader()
            print(f'    dataset size is {len(train_dataset)}')

            # initialize learner and optimizer
            model = PGEN_NN().to(device)
            optimizer = optim.Adadelta(model.parameters())

            name = f'pgen_nn_episode_{i}_time_{t}_action_{at}'
            train_model(model, optimizer, args, train_loader, device, cnn, name)  # train learner
            loss_val, et = test(model, device, dev_loader, cnn, verbos=False)  # evaluate learner
            # hold data
            error.append(et)
            loss.append(loss_val)
            print(f'    error of the learner was {et}')
            rt = e0 - et  # calculate reward
            # optimization step
            if q_first[st, at] == 0:
                Q[st, at] = rt
                q_first[st, at] = 1
            else:
                Q[st, at] += alpha * (rt + gamma * (Q[at].max()) - Q[st, at])
            st = at
            e0 = et
            t += 1
            print('--------------')
        errors.append(np.array(error))
        losses.append(np.array(loss))

        if args.save_model:
            np.save(PATH + 'policy', Q)
            np.save(PATH + 'q_first', q_first)
            np.save(PATH + 'errors', errors)
            np.save(PATH + 'losses', losses)

        end_time = time.time()
        print('Episode took %.3f seconds' % (end_time - start_time))

    return Q, errors, losses


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
    parser.add_argument('--n-episode', type=int, default=600, metavar='N',
                        help='Number of examples in each episode')
    parser.add_argument('--discount-factor', type=float, default=0.9, metavar='N',
                        help='Discount factor for the curiosity loop')
    parser.add_argument('--threshold', type=float, default=-0.05, metavar='N',
                        help='Threshold for Q table')
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

    # convert to numpy for later processing
    train_images = np.array([dataset1[i][0].numpy() for i in range(len(dataset1))])
    train_labels = np.array([dataset1[i][1] for i in range(len(dataset1))])

    # define, load and freeeze the classifying CNN
    cnn = Net().to(device)
    cnn.load_state_dict(torch.load(args.cnn_path))
    for param in cnn.parameters():
        param.requires_grad = False
    cnn.eval()

    _, _, _ = curiosity_loop(train_images, train_labels, args, device, cnn)


if __name__ == '__main__':
    main()
