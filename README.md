Curiosity_project
Part 3 of the project - Curiosity Loop
Note that some function assume a folder called trained_models was created.

Important files:
- mnist_model.py - creates and trains a CNN for classification of handwritten digits (MNIST dataset)
	Note that a trained model mnist_cnn_epoch62.pt is supplied under the trained_models directory.
- NN_pert_gen.py - creates and trains a CNN for generation of adversarial examples (AE) based on input images.
- curiosity_loop.py - runs a curiosity loop and tests it.
- curiosity_loop_analysis.py - creates graph and compares learned policy with random policy.
- dataset.py - a Dataset object for inner use in curiosiy loop and related algorithms.

All the code assumes a windows operating system.

To run start with curiosity_loop.py and proceed to curiosity_loop_analysis.py.
Running this may take over 40 hours (using i7 7700k processor and nvidia gtx 1080ti gpu)
Note that under the trained_models folder you can find all the files creted by running
these, except for trained models which are not required for the analysis.

some packages needed for running are torch, torchvision, numpy, matplotlib, scipy.