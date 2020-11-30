import pandas as pd
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import math
from mnist_model import Net

EPS = 0.25 #change perturbation makes
REPS  = 6 #repetitions of process per sample
BATCH_SIZE = 1000 #batch-size for reprediction
PIC_DIM = 28 #the dimensions of the samples (28 by 28)
B_SIZE = 1

def break_data(data):
    """
    input:
    - data: torch tensor containing both pictures and labels
    -----
    output:
    - test_list: numpy array of the pictures
    - test_labels: numpy array of the labels
    """
    test_list = np.array([data[i][0].numpy() for i in range(len(data))])
    test_labels = np.array([data[i][1] for i in range(len(data))])
    
    return test_list, test_labels

def create_batches(pictures, labels, batch_size):
    """
    input:
    - pictures: numpy array of the pictures
    - labels: numpy array of the labels
    - batch_size: int, size of the batches
    -----
    output:
    - batches: list containing batches each batch is a tuple of torch's Variable of tensors:
        1) The input features
        2) The corresponding labels
    """
    batches = []
    cur_batch = 0
    number_of_batches = math.ceil(len(pictures) / batch_size)
    
    while cur_batch < number_of_batches:
       batch = (pictures[cur_batch * batch_size: (cur_batch + 1) * batch_size], labels[cur_batch * batch_size: (cur_batch + 1) * batch_size]) #slice
       batch = (torch.from_numpy(batch[0]), torch.from_numpy(batch[1]).long()) #to tensor
       batch = (Variable(batch[0], requires_grad = True), Variable(batch[1])) #to Variable
       cur_batch += 1
       batches.append(batch)
       
    return batches

def get_predictions(pictures, labels, model, device, batch_size = BATCH_SIZE):
    """
    input:
    - pictures: numpy array of the pictures
    - labels: numpy array of the labels
    - model: the CNN to predict the new examples' label
    - device: torch device to use in the model while predicting
    - batch_size: int, default = BATCH_SIZE, size of the batches
    -----
    output:
    - predictions: numpy array of the predictions of each of the samples in the pictures array
    - correct: int number of correct predictions
    - correct_inds_set: set of indices of correctly classified examples (if used make sure to set batch_size to 1)
    - incorrect_inds_set: set of indices of incorrectly classified examples (if used make sure to set batch_size to 1)
    """
    model.eval() #so we won't apply dropout layers
    correct = 0
    batches = create_batches(pictures, labels, batch_size) #create batches
    predictions = None
    
    correct_inds_set = set() #used for the alternative discussed in the appendix of the attached document, use batch_size = 1 for this
    incorrect_inds_set = set()
    
    with torch.no_grad():
        for i, sample in enumerate(batches):
            x, y = sample
            x, y = x.to(device), y.to(device)
            
            #get predictions
            output = model(x)
            pred = output.argmax(dim = 1, keepdim = True)
            
            if predictions is None:
                predictions = pred.numpy()
            else:
                predictions = np.concatenate((predictions, pred.numpy()))
                
            temp = pred.eq(y.view_as(pred)).sum().item() #count correct predictions
            correct += temp

            #update sets
            if temp == 1:
                correct_inds_set.add(i)
            else:
                incorrect_inds_set.add(i)
            
    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(pictures),
        100. * correct / len(pictures)))
    
    return predictions, correct, correct_inds_set, incorrect_inds_set

def save_data(s, a, x, path, rep):
    """
    input:
    - s: numpy array of the true labels of the pictures
    - a: numpy array of the actions applied pn the pictures 
    - x: numpy array of the predicted labels of the pictures
    - path: path to which save the data
    - rep: int, the repetition number
    -----
    the functions saves triplets of (si, ai, xi) to a csv in path.
    if rep is 0 (first repetition) creates the csv, o/w adds to it.
    """
    data = np.concatenate((s, a, x), axis = 1)
    df = pd.DataFrame(data = data)
    if rep == 0: #first repetition, need to create csv
        df.to_csv(path_or_buf = path + ".csv", index = False, header = ['s', 'a', 'x']) # save to csv
    else: #csv was created, need to add to it, without header
        df.to_csv(path_or_buf = path + ".csv", index = False, header = False, mode = 'a') # save to csv

def fix_arrays(actions):
    """
    input:
    - actions: numpy array of the actions attributed to each picture, while the first dimension length is equal
        to the number of samples, the second dimension contains varying sizes.
    -----
    output:
    - indices: numpy array of integers used for repeating a sample to fit the number of acions attributed to it
    """
    indices = []
    for i, picture in enumerate(actions):
        indices += [i for action in picture]
    return indices

def noise2actions(noise):
    """
    input:
    - noise: numpy array representing the noise (or perturbations) added to the samples
    -----
    output:
    - actions: numpy array of integers with two dimensions. The first dimension fits the samples, and the second contains enumeration
        of the actions atributed to each sample. Note that the second dimension is not of constant size. For the rules of enumeration please
        refer to the attached document.
    """
    actions = []
    
    for picture in noise:
        flat = picture.flatten()
        action = []
        #add enumeration
        for i, cell in enumerate(flat):
            if cell > 0: #added epsilon
                action.append(2 * i + 1)
            elif cell < 0: #subtructed epsilon
                action.append(2 * i + 2)
        actions.append(np.array(action))
        
    return np.array(actions)

def add_noise(data, b_size = 1):
    """
    input:
    - data: numpy array containing the samples (pictures)
    - b_size: integer (default = 1) represnting the block size for the noise resolution
    -----
    output:
    - new_data: numpy array of of the samples with the added noise.
    - noise_actions: numpy array containing the actions attributed to each sample, note that the
        second dimension is of varying size
    """
    new_data = data.copy()
    noise_actions = None
    
    if b_size == 1: #no blocks - 1:1 noise to pixel 
        s_noise = np.random.randint(-1, high = 2, size = data.shape) #generate perturbation
        s_noise = s_noise * EPS
        noise_actions = noise2actions(s_noise)
    else: #using blocks
        shape = data.shape
        noise_shape = (shape[0], shape[1], shape[2] // b_size, shape[3] // b_size)
        s_noise = np.random.randint(-1, high = 2, size = noise_shape) #generate perturbation
        s_noise = s_noise * EPS
        noise_actions = noise2actions(s_noise)
        s_noise = np.repeat(np.repeat(s_noise, b_size, axis = 3), b_size, axis = 2) #transform the noise to size of the pictures
    new_data += s_noise
    new_data = np.clip(new_data, 0, 1) #clip to original range
    
    return new_data , noise_actions

def manual_flatten(array):
    """
    input:
    - array: numpy array with second dimension of varying sizes
    -----
    output:
    - res: numpy array like the flattened input array. This function was required
        since numpy can't work with varying sizes
    """
    res = array[0]
    for a in array[1:]:
        res = np.concatenate((res, a))
    res = np.expand_dims(res, axis = 1)
    return res

def gen_data(pictures, labels, path, model, device, org_cor_set):
    """
    input:
    - pictures: numpy array of the pictures
    - labels: numpy array of the true labels of the pictures
    - path: path to save csv to
    - model: CNN model to predict perturbated pictures
    - device: torch device to run the model on
    - org_cor_set: set representing the indices of correctly classified original examples
    -----
    The function creates perturbation of the pictures, repredicts them using the model
    and then saves the information to a csv file in path.
    """
    print('breaking data...')
    total_corret = 0
    total_examples = 0
    difference = 0
    
    print('starting loop...')
    for rep in range(REPS):
        print('starting repetition %d...' % (rep + 1))
        new_pictures, actions = add_noise(pictures, b_size = B_SIZE)
        predictions, correct, cor_set, incor_set = get_predictions(new_pictures, labels, model, device)
        total_corret += correct
        total_examples += len(new_pictures)
        #uncomment the following lines for the alternative discussed in the appendix of the attached document
        #relevant = list(incor_set.intersection(org_cor_set))
        #labels_copy, predictions, actions = labels[relevant], predictions[relevant], actions[relevant]
        indices = fix_arrays(actions)
        save_data(np.expand_dims(labels[indices], axis = 1), manual_flatten(actions), predictions[indices], path, rep) #replace this line with the following if using alternative
        #save_data(np.expand_dims(labels_copy[indices], axis = 1), manual_flatten(actions), predictions[indices], path, rep)
    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
        total_corret, total_examples,
        100. * total_corret / total_examples))


if __name__ == '__main__':
    #define paths
    model_path = 'trained_models\mnist_cnn_epoch62.pt'
    csv_path = 'data\generated_data'
    indices_path = 'data\indices.npy'

    #get dataset
    print('downloading dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST('../data', train = False, download = True,
                       transform = transform)

    #load trained model
    print('creating model...')
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))

    #split data to separate pictures and labels arrays
    print('splitting data...')
    pictures, labels = break_data(dataset)

    #randomize order and split to two datasets
    if not os.path.exists(indices_path):
        indices = np.arange(len(pictures))
        np.random.shuffle(indices)
        np.save(indices_path, indices)
    else:
        indices = np.load(indices_path)

    pictures, labels = pictures[indices], labels[indices]
        
    pictures, labels = pictures[:len(pictures) // 2], labels[:len(labels) // 2]

    #get baseline
    _, _, cor_set, incor_set = get_predictions(pictures, labels, model, device, batch_size = 1)

    #generate data
    gen_data(pictures, labels, csv_path, model, device, cor_set)
                        
