from torch.utils.data import Dataset
import numpy as np

class ShuffleDataLoader(Dataset):
    def __init__(self, images, labels):

        shuffled_indices = np.random.permutation(len(labels))

        self.images = [images[index] for index in shuffled_indices]
        self.labels = [labels[index] for index in shuffled_indices]


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label