import pandas as pd
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import math
from mnist_model import Net

EPS = 0.25
REPS  = 6
BATCH_SIZE = 1000

def split_data(data):
    test_list = np.array([data[i][0].numpy() for i in range(len(data))])
    test_labels = np.array([data[i][1] for i in range(len(data))])
    return test_list, test_labels

def create_batches(pictures, labels, batch_size):
    """
    This function should recieve the data, and return a list of batches, each batch is a tuple
    of torch's Variable of tensors:
    1) The input features
    2) The corresponding labels
    """
    batches = []
    cur_batch = 0
    number_of_batches = math.ceil(len(pictures) / batch_size)
    while cur_batch < number_of_batches:
       batch = (pictures[cur_batch * batch_size: (cur_batch + 1) * batch_size], labels[cur_batch * batch_size: (cur_batch + 1) * batch_size])
       batch = (torch.from_numpy(batch[0]), torch.from_numpy(batch[1]).long())
       batch = (Variable(batch[0], requires_grad = True), Variable(batch[1]))
       cur_batch += 1
       batches.append(batch)
    return batches


def get_predictions(pictures, labels, model, device):
    model.eval()
    correct = 0
    batches = create_batches(pictures, labels, BATCH_SIZE)
    predictions = None
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim = 1, keepdim = True)
            if predictions is None:
                predictions = pred.numpy()
            else:
                predictions = np.concatenate((predictions, pred.numpy()))
            correct += pred.eq(y.view_as(pred)).sum().item()
    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(pictures),
        100. * correct / len(pictures)))
    return predictions, correct

def save_data(s, a, x, path, rep):
    """
    input:
    - s, numpy array of ints shape (n * a, 1): representing the prediction of the original photo.
        On the first n elements a1 was applied, on the second n elements a2 was apllied and s on  
    - a, int: the number of actions (no need for actual actions as there is 1:1 enumeration)
    - x, numpy array of ints shape (n * a, 1): representing the new prediction after the added noise
    - path, string: path to save data to
    - rep, int: indicates the repetition
    -----------
    The function saves the examples to the path. If the path exsits adds the data to it, o/w creates a csv file
    """

    actions = np.arange(1, a + 1, dtype = int)
    n = s.shape[0]
    actions = np.repeat(actions,n)
    examples = np.asarray([s, actions, x])
    if rep == 0:
        if not os.path.isfile(path):
            f = np.savetxt(path + ".csv", examples, delimiter = ",")
            return f
        else:
            raise Exception("File already exists")
    elif rep > 0:
        with open(path, 'wb') as f:
            f = np.savetxt(f, examples, delimiter = ",")
        return f
    else:
        raise Exception("rep argument must be a positive number")

def save_data_v2(s, a, x, path, rep):
    data = np.concatenate((s, a, x), axis = 1)
    df = pd.DataFrame(data = data)
    if rep == 0:
        if not os.path.isfile(path):
            df.to_csv(path_or_buf = path + ".csv", index = False, header = ['s', 'a', 'x']) # save to csv
        else:
            raise Exception("File already exists")
    elif rep > 0:
        df.to_csv(path_or_buf = path + ".csv", index = False, header = ['s', 'a', 'x'], mode = 'a') # save to csv
    else:
        raise Exception("rep argument must be a positive number")

def fix_arrays(actions):
    indices = []
    for i, picture in enumerate(actions):
        indices += [i for action in picture]
    return indices

def get_actions():
    actions = set()
    ret = []
    pos = [i / 20 for i in range(21)]
    for a in pos:
        for b in pos:
            if a + b > 1:
                continue
            else:
                temp = (a, b, round(1 - a - b, 2))
                if temp not in actions:
                    actions.add(temp)
                    ret.append(temp)
    return ret

def add_noise(action, data):
    new_data = data.copy()
    a, b, c = action
    g_noise = np.random.normal(loc = 0, scale = EPS, size = data.shape)
    new_data += a * g_noise
    u_noise = np.random.rand(data.shape[0], data.shape[1], data.shape[2], data.shape[3])
    u_noise = u_noise * 2 * EPS - EPS
    new_data += b * u_noise
    s_noise = np.random.randint(-1, high = 2, size = data.shape)
    s_noise = s_noise * EPS
    new_data += c * s_noise    
    return new_data

def noise2actions(noise):
    actions = []
    for picture in noise:
        flat = picture.flatten()
        action = []
        for i, cell in enumerate(flat):
            if cell > 0:
                action.append(2 * i + 1)
            elif cell < 0:
                action.append(2 * i + 2)
        actions.append(np.array(action))
    return np.array(actions)

def add_noise_v2(data):
    new_data = data.copy()
    s_noise = np.random.randint(-1, high = 2, size = data.shape)
    s_noise = s_noise * EPS
    noise_actions = noise2actions(s_noise)
    new_data += s_noise
    return np.clip(new_data, 0, 1), noise_actions

def show_noisy_data(action, data, path):
    print(action)
    new_data = data.copy()
    a, b, c = action
    g_noise = np.random.normal(loc = 0, scale = EPS, size = data.shape)
    new_data += a * g_noise
    u_noise = np.random.rand(data.shape[0], data.shape[1], data.shape[2], data.shape[3])
    u_noise = u_noise * 2 * EPS - EPS
    new_data += b * u_noise
    s_noise = np.random.randint(-1, high = 2, size = data.shape)
    s_noise = s_noise * EPS
    new_data += c * s_noise

    new_data = [torch.from_numpy(picture) for picture in new_data]
    org_data = [torch.from_numpy(picture) for picture in data]

    torchvision.utils.save_image(torchvision.utils.make_grid(new_data[:16]), path + '_noisy' + '.jpg')
    torchvision.utils.save_image(torchvision.utils.make_grid(org_data[:16]), path + '_original' + '.jpg')

def gen_data(data, path, model, device):
    print('breaking data...')
    pictures, labels = split_data(data)
    actions = get_actions()
    total_corret = 0
    print('starting loop...')
    for rep in range(REPS):
        new_labels = np.tile(labels, len(actions))
        predictions = None
        is_first = True
        for i, action in enumerate(actions):
            #show_noisy_data(action, pictures, 'image')
            print('rep %d, action %d' % (rep + 1, i + 1))
            temp = add_noise(action, pictures)
            if not is_first:
                temp_preds, correct = get_predictions(temp, labels, model, device)
                predictions = np.concatenate((predictions, temp_preds))
                total_corret += correct
            else:
                predictions, correct = get_predictions(temp, labels, model, device)
                total_corret += correct
                is_first = False
        print('Accuracy: {}/{} ({:.0f}%)\n'.format(
            total_corret, len(pictures) * len(actions),
            100. * total_corret / (len(pictures) * len(actions))))
        save_data(new_labels, len(actions), predictions, path, rep)

def manual_flatten(array):
    res = array[0]
    for a in array[1:]:
        res = np.concatenate((res, a))
    return np.expand_dims(res, axis = 1)

def gen_data_v2(data, path, model, device):
    print('breaking data...')
    pictures, labels = split_data(data)
    total_corret = 0
    total_examples = 0
    print('starting loop...')
    for rep in range(REPS):
        print('starting repetition %d...' % (rep + 1))
        new_pictures, actions = add_noise_v2(pictures)
        predictions, correct = get_predictions(new_pictures, labels, model, device)
        total_corret += correct
        total_examples += len(new_pictures)
        indices = fix_arrays(actions)
        save_data_v2(np.expand_dims(labels[indices], axis = 1), manual_flatten(actions), predictions[indices], path, rep)
    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_corret, total_examples,
        100. * total_corret / total_examples))
          

def main(model_path, csv_path):
    
    print('downloading dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST('../data', train = False, download = True,
                       transform = transform)

    print('splitting dataset...')
    test_set, _ = torch.utils.data.random_split(dataset, [5_000, 5_000])

    print('creating model...')
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))
    gen_data_v2(test_set, csv_path, model, device)

main('trained_models\mnist_cnn_epoch62.pt', 'generated_data')
                        
