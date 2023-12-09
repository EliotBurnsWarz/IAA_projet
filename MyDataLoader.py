from torch.utils.data import Dataset
import numpy as np
import random

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
    
    
class OrderedDataSet():
    def __init__(self, images):
        self.images = images

    def get_same_class(self):
        class_num = random.randint(0, len(self.images) - 1)
        class_data = self.images[class_num]
        
        im1_index = random.randint(0, len(class_data)-1)
        im2_index = random.randint(0, len(class_data)-1)
            
        while im2_index == im1_index:
            im2_index = random.randint(0, len(class_data)-1)
            
        image1 = class_data[im1_index]
        image2 = class_data[im2_index]

        return image1, image2

    def get_diff_class(self):
        class_num1 = random.randint(0, len(self.images)-1)
        class_num2 = random.randint(0, len(self.images) - 1)

        while class_num2 == class_num1:
            class_num2 = random.randint(0, len(self.images) - 1)
        
        class_data1 = self.images[class_num1]
        class_data2 = self.images[class_num2]
        
        image1 = class_data1[random.randint(0, len(class_data1)-1)]
        image2 = class_data2[random.randint(0, len(class_data2)-1)]
        
        return image1, image2

        


class SiameseDataLoader(Dataset):
    def __init__(self, ordered_dataset, batches_per_epoch):

        self.ordered_dataset = ordered_dataset
        self.batches_per_epoch = batches_per_epoch
        
    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self):
        same_class = random.choice([1, 0], weights=[0.9, 0.1])
        if same_class:
            im1, im2 = self.ordered_dataset.get_same_class()
            
        else: 
            im1, im2 = self.ordered_dataset.get_diff_class()
        
        return im1, im2, same_class