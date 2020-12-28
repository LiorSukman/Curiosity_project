Curiosity_project
Part 2 of the project - Neural Network
Note that some function assume a folder called trained_models and illustrations_and_results were created.

Important files:
- mnist_model.py - creates and trains a CNN for classification of handwritten digits (MNIST dataset)
	Note that a trained model mnist_cnn_epoch62.pt is supplied under the trained_models directory
- NN_pert_gen.py - creates and trains a CNN for generation of adversarial examples (AE) based on input images
- model_comparison.ipynb - a jupyter notebook containing all comparisons and visualizations required for this part
	(it is also available as html file for easier reading). Please note that running this notebook takes a long
	time and so a use of cuda is advaisable.

All the code assumes a windows operating system.

some packages needed for running are torch, torchvision, numpy, matplotlib, pandas, jupyter, tabulate.