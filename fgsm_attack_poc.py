import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from mnist_model import Net
import matplotlib.pyplot as plt

EPS = [0, .05, .1, .15, .2, .25, .3]
BATCH_SIZE = 1
PIC_DIM = 28

# The code in this file was adapted from the notebook that can be found here:
# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

"""
This function is similar to combine_grad_val but it uses the signs instead of the gradient
values. Hence, this approach is much slower (perhaps as a result of imperfect implementation)

def combine_grad_sign(grad):
    new_grad = grad.detach().clone()
    new_grad.requires_grad = False
    partition = 4
    for i in range(PIC_DIM // partition):
        for j in range(PIC_DIM // partition):
            total_0 = 0
            total_p = 0
            total_n = 0
            for k in range(partition):
                for l in range(partition):
                    kl_grad = grad[0][0][i * partition + k][j * partition + l]
                    total_0 += 1 if kl_grad == 0 else 0
                    total_p += 1 if kl_grad == 1 else 0
                    total_n += 1 if kl_grad == -1 else 0
            sign = None
            if max(total_0, max(total_p, total_n)) == total_0:
                sign = 0
            elif max(total_p, total_n) == total_p:
                sign = 1
            else:
                sign = -1
            for k in range(partition):
                for l in range(partition):
                    new_grad[0][0][i * partition + k][j * partition + l] = sign
            
    return new_grad
"""

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""
    return (array.reshape(1, 1, PIC_DIM // nrows, nrows, -1, ncols)
                 .swapaxes(3, 4)
                 .reshape(1, 1, -1, nrows, ncols))

def combine_grad_val(grad):
    b_rows = 2 #number of rows in block
    b_cols = 2 #number of columns in block
    new_grad = split(grad.numpy(), b_rows, b_cols)
    new_grad = np.average(new_grad, (3, 4)).reshape(1, 1, PIC_DIM // b_rows, PIC_DIM // b_cols) #the average gradient in each block
    new_grad = torch.from_numpy(np.repeat(np.repeat(new_grad, b_cols, axis = 3), b_rows, axis = 2)) #back to original dimensions
    return new_grad.sign() #return sign

kinds = ['full', 'combined_sign', 'combined_val']

def fgsm_attack(image, epsilon, data_grad, kind = 'full'):
    if epsilon == 0:
        return image
    # Collect the element-wise sign of the data gradient
    if kind == 'full':
        sign_data_grad = data_grad.sign()
    elif kind == 'combined_sign':
        sign_data_grad = combine_grad_sign(data_grad.sign())
    else:
        sign_data_grad = combine_grad_val(data_grad)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test_eps(model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim = True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad, kind = 'combined_val')

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim = True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def plot_graphs(accuracies, examples):
    plt.figure(figsize = (5, 5))
    plt.plot(EPS, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step = 0.1))
    plt.xticks(np.arange(0, .35, step = 0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize = (8, 10))
    for i in range(len(EPS)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(EPS), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(EPS[i]), fontsize = 14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap = "gray")
    plt.tight_layout()
    plt.show()

def main(model_path):
    
    print('downloading dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST('../data', train = False, download = True,
                       transform = transform)

    test_kwargs = {'batch_size': BATCH_SIZE}
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    print('creating model...')
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in EPS:
        acc, ex = test_eps(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    plot_graphs(accuracies, examples)
    

main('trained_models\mnist_cnn_epoch62.pt')
                        
