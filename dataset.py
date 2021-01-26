import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    """
        A class used to represent a dataset

        Attributes
        ----------
        images : torch tensor
            torch tensor of the images
        labels : torch tensor
            torch tensor of the labels corresponding to the images tensor
        kwargs : dictionary
            dictionary containing the parameters for creation of dataloader

        Methods
        -------
        get_dataloader(self)
            Returns a torch dataloader using the data and according to the kwargs
        """
    def __init__(self, images, labels, classes, batch_size, use_cuda):
        """
        initializes a Dataset object

            Parameters
            ----------
            images : numpy array
                numpy array of the images
            labels : numpy array
                numpy array of the labels corresponding to the images tensor
            classes : list of integers
                list containing the classes to keep in the dataset
            batch_size : int
                the batch size to use when creating a dataloader
            use_cuda : bool
                whether to use cuda (GPU) in the dataloader

            """
        if classes is not None:
            indices = [(labels == c).nonzero()[0] for c in classes]  # keep only required classes
            indices = np.hstack(tuple(indices))
        else:
            indices = np.arange(len(labels))

        self.images = torch.tensor(images[indices])
        self.labels = torch.tensor(labels[indices]).long()
        self.kwargs = {'batch_size': batch_size, 'shuffle': True}

        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True}
            self.kwargs.update(cuda_kwargs)

    def __len__(self):
        """
        calculates and returns the number of examples in the dataset
            Returns
            -------
            int
                the number of examples in the dataset
        """
        return len(self.labels)

    def __getitem__(self, item):
        """
        finds and returns the item in given location of the dataset

            Parameters
                ----------
                item : integer or a list of integers
                    the indices to retrieve

            Returns
            -------
            torch tensor, torch long
                an example in specified location given by image and label
        """
        return self.images[item], self.labels[item]

    def get_dataloader(self):
        """
        creates and returns torch dataloader of the dataset

            Returns
            -------
            torch dataloadaer
                dataloader of the data using kwargs (batch size, shuffle, num_workers, pin_memory)
        """
        return torch.utils.data.DataLoader(self, **self.kwargs)
