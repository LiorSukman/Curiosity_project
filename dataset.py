import numpy as np
import torch


class Dataset:
    def __init__(self, images, labels, classes, batch_size, use_cuda):
        if classes is not None:
            indices = [(labels == c).nonzero()[0] for c in classes]
            indices = np.hstack(tuple(indices))
        else:
            indices = np.arange(len(labels))

        self.data = [(images[i], labels[i]) for i in indices]
        self.kwargs = {'batch_size': batch_size}

        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            self.kwargs.update(cuda_kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_dataloader(self):
        return torch.utils.data.DataLoader(self.data, **self.kwargs)
