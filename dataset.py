import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, classes, batch_size, use_cuda):
        if classes is not None:
            indices = [(labels == c).nonzero()[0] for c in classes]
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
        return len(self.labels)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]

    def get_dataloader(self):
        return torch.utils.data.DataLoader(self, **self.kwargs)