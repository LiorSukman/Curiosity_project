import pandas as pd
import numpy as np
import torch
import scipy
import os

EPS = 0.05
REPS  = 5

def split_data(data):
    raise NotImplemented

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

    actions = np.arange(1, a + 1, dtype=int)
    n=s.shape[0]
    actions=np.repeat(actions,n)
    examples = np.asarray([s, actions, x])
    if rep == 0:
        if not os.path.isfile(path):
            f=np.savetxt(path + ".csv", examples, delimiter=",")
            return f
        else:
            raise Exception("File already exists")
    elif rep > 0:
        with open(path, 'wb') as f:
            f=np.savetxt(f, examples, delimiter=",")
        return f
    else:
        raise Exception("rep argument must be a positive number")





def get_predictions(x, y, model):
    raise NotImplemented

def get_actions():
    actions = set()
    ret = []
    pos = [i / 20 for i in range(21)]
    print(pos)
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
    a, b, c = action
    g_noise = np.random.normal(loc = 0, scale = EPS, size = data.shape)
    u_noise = np.random.rand(data.shape) # need every dimension separately
    u_noise = u_noise * 2 * EPS - EPS
    s_noise = np.random.randint(-1, high = 1, size = data.shape)
    s_noise = s_noise * EPS
    return data + a * g_noise + b * u_noise + c * s_noise

def gen_data(data, path, model):
    pictures, labels = split_data(data)
    actions = get_actions()
    for rep in range(REPS):
        new_pictures = None
        new_labels = np.tile(labels, len(actions))
        predictions = None
        is_first = True
        for action in actions:
            if not is_first:
                temp = add_noise(action, pictures)
                new_pictures = np.concatenate((new_pictures, temp))
                predictions = np.concatenate((predictions, get_predictions(temp, labels, model)))
            else:
                new_pictures = add_noise(action, pictures)
                predictions = get_predictions(temp, labels, model)
                is_first = False
        save_data(new_labels, len(actions), predictions, path, rep)

print(len(get_actions()))
                        




