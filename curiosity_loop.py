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


def sample_episode(episode_size, train_images, train_labels, classes=np.arange(10)):
    sample_size = episode_size // len(classes)
    indices = [np.nonzero(train_labels == c)[0] for c in classes]

    images = []
    labels = []
    for inds in indices:
        meta_inds = np.random.choice(inds.flatten(), size=sample_size, replace=False)
        images.append(train_images[meta_inds])
        labels.append(train_labels[meta_inds])

    images = np.vstack(tuple(images))  # TODO recheck this stacking, something is weird that they are not the same
    labels = np.hstack(tuple(labels))

    return images, labels


def split_data(x, y, percentage, classes):
    percentage = 1 - percentage
    indices = [np.nonzero(y == c)[0] for c in classes]
    x_train, x_test, y_train, y_test = [], [], [], []

    for inds in indices:
        np.random.shuffle(inds)
        x_train.append(x[inds[:int(percentage * len(inds))]])
        x_test.append(x[inds[int(percentage * len(inds)):]])
        y_train.append(y[inds[:int(percentage * len(inds))]])
        y_test.append(y[inds[int(percentage * len(inds)):]])

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.hstack(y_train)
    y_test = np.hstack(y_test)

    return x_train, x_test, y_train, y_test


def train_model(model, optimizer, args, dataloader, device, cnn, name):
    # TODO check if using train set to stop learning is methodologically ok
    best_epoch = 0
    best_loss = float('inf')
    model_name = name + '_best.pt'
    # losses = []
    start_time = time.time()
    # run training
    for epoch in range(1, args.epochs + 1):
        if epoch % 10 == 0:
            print(f'        In epoch {epoch}')
        train(args, model, device, dataloader, optimizer, epoch, cnn, verbos=False)
        loss, accuracy = test(model, device, dataloader, cnn, verbos=False)
        # losses.append(loss)
        if loss < best_loss:  # found better epoch
            best_loss = loss
            best_epoch = epoch
            torch.save(model.state_dict(), PATH + model_name)

    end_time = time.time()
    print('    Training took %.3f seconds' % (end_time - start_time))
    print('    Best model was achieved on epoch %d' % best_epoch)
    model.load_state_dict(torch.load(PATH + model_name))  # load model from best epoch

    # plt.plot(np.arange(1, args.epochs + 1), losses)
    # plt.show()


def curiosity_loop(train_images, train_labels, args, device, cnn, classes=np.arange(10)):
    # TODO move constants to args
    N_episode = 600
    N_iter = 10 * train_images.shape[0] // N_episode
    gamma = 0.01

    if args.load_model:
        Q = np.load(args.load_path + 'policy.npy')
        errors = list(np.load(args.load_path + 'errors.npy'))
        losses = list(np.load(args.load_path + 'losses.npy'))
        cur_episode = len(losses)

        random.seed(args.seed + cur_episode)
        np.random.seed(args.seed + cur_episode)
        torch.manual_seed(args.seed + cur_episode)
    else:
        Q = np.ones((len(classes) + 1, len(classes)))  # TODO need to rethink value if using loss instead of error rate
        errors = []
        losses = []
        cur_episode = 0

    for i in range(cur_episode, N_iter):
        start_time = time.time()
        print(f'************Starting episode {i + 1}************')
        episode_images, episode_labels = sample_episode(N_episode, train_images, train_labels, classes=classes)
        train_episode_images, dev_episode_images, train_episode_labels, dev_episode_labels = \
            split_data(episode_images, episode_labels, 0.2, classes)
        e0 = 0.015  # TODO rethink this value - should probably be the training error rate
        st = 10
        c_sel = []
        c_available = list(classes)
        eps = 0.9 if i < N_iter / 4 else (0.5 if (N_iter / 4 <= i < N_iter / 2) else
                                          (0.3 if (N_iter / 2 <= i < 3 * N_iter / 4) else 0.1))
        alpha = 0.09 if i < N_iter / 4 else (0.05 if (N_iter / 4 <= i < N_iter / 2) else
                                             (0.01 if (N_iter / 2 <= i < 3 * N_iter / 4) else 0.005))
        dev_dataset = Dataset(dev_episode_images, dev_episode_labels, None, args.test_batch_size, use_cuda)
        dev_loader = dev_dataset.get_dataloader()
        t = 0
        error = []
        loss = []
        while len(c_available) != 0:
            print(f'    Starting time step {t}')
            if eps > random.random():
                at = random.sample(c_available, 1)[0]
            else:
                qa = Q[st].copy()
                qa[c_sel] = 0
                if qa.max() <= 0:
                    break
                at = qa.argmax()
            print(f'    rl chose action {at}')
            c_available.remove(at)
            c_sel.append(at)
            train_dataset = Dataset(train_episode_images, train_episode_labels, c_sel, args.batch_size, use_cuda)
            train_loader = train_dataset.get_dataloader()
            print(f'    dataset size is {len(train_dataset)}')

            model = PGEN_NN().to(device)
            optimizer = optim.Adadelta(model.parameters())

            name = f'pgen_nn_episode_{i}_time_{t}_action_{at}'
            train_model(model, optimizer, args, train_loader, device, cnn, name)
            loss_val, acc = test(model, device, dev_loader, cnn, verbos=False)
            et = 1 - acc
            error.append(et)
            loss.append(loss_val)
            print(f'    error rate of the learner was {et}')
            rt = et - e0  # TODO reconsider as I flipped it
            Q[st, at] += alpha * (rt + gamma * (Q[at].max()) - Q[st, at])
            st = at
            e0 = et
            t += 1
            print('--------------')
        errors.append(np.array(error))
        losses.append(np.array(loss))

        if args.save_model:
            np.save(PATH + 'policy', Q)
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

    policy, errors, losses = curiosity_loop(train_images, train_labels, args, device, cnn)


if __name__ == '__main__':
    main()
