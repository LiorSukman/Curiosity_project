import torch
import torch.nn as nn
import torch.nn.functional as F
from NN_pert_models import PGEN_NN1
from NN_pert_models import PGEN_NN2
from NN_pert_models import PGEN_NN3
import numpy as np

sample_size = 50_000

def number_of_params(model):
    """
    :inputs: cnn model
    :return number of weights in the model
    :rtype: int
    """
    params = model.parameters()
    count = 0
    for param in params:
        num_params = list(torch.reshape(param, (-1,)).size())[0]
        count += num_params
    return count

def calculate_BIC(model, nn_avg_lose):
    """
    Calculate the Bayesian Information Criterion (BIC) for a GPy `model` with maximum likelihood hyperparameters on a given dataset
    https://enwikipediaorg/wiki/Bayesian_information_criterion
    """
    # nn_avg_lose is the natural logarithm of the marginal likelihood of the Gaussian process
    # sample_size is the number of data points
    # number_of_params() is the number of optimisation parameters
    return - 2 * np.log(nn_avg_lose) + sample_size * np.log(number_of_params(model))

def calculate_AIC(nn_avg_lose):
    # nn_avg_lose is the natural logarithm of the marginal likelihood of the Gaussian process
    # sample_size is the number of data points
    return - 2 * np.log(nn_avg_lose) + 2 * sample_size

def main():
    """"
    trainning and testing executed on Google Colab:
    https://colab.research.google.com/drive/1uQK0Wb331hbEj5Eh3UW-K4OOuYIRXI8s?usp=sharing
    """
    # nn1_best_epoch = "trained_models_for_compare/trained_models_PGEN_NN1_epoch30.pt"
    # nn2_best_epoch = "trained_models_for_compare/trained_models_PGEN_NN2_epoch34.pt"
    # nn3_best_epoch = "trained_models_for_compare/trained_models_PGEN_NN3_epoch35.pt"

    nn1_avg_lose = 0.7041
    nn2_avg_lose = 1.260
    nn3_avg_lose = 1.2081

    nn1 = PGEN_NN1()
    nn2 = PGEN_NN2()
    nn3 = PGEN_NN3()

    print("nn1 AIC: ", calculate_AIC(nn1_avg_lose))
    print("nn2 AIC: ", calculate_AIC(nn2_avg_lose))
    print("nn3 AIC: ", calculate_AIC(nn3_avg_lose))

    print("nn1 BIC: ", calculate_BIC(nn1, nn1_avg_lose))
    print("nn2 BIC: ", calculate_BIC(nn2, nn2_avg_lose))
    print("nn3 BIC: ", calculate_BIC(nn3, nn3_avg_lose))


if __name__ == '__main__':
    main()

