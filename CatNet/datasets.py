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

    def __len__(self):
        return len(self.images)

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
    def __init__(self, ordered_dataset, batches_per_epoch, ratio = [0.25, 0.75]):

        self.ordered_dataset = ordered_dataset
        self.batches_per_epoch = batches_per_epoch

        self.ratio = ratio
        
    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self):
        # same_class = random.choice([1, 0], weights=[0.9, 0.1])
        same_class = random.choice([1, 0], weights = self.ratio)
        if same_class:
            im1, im2 = self.ordered_dataset.get_same_class()
            
        else: 
            im1, im2 = self.ordered_dataset.get_diff_class()
        
        return im1, im2, same_class
  

class SiameseDataset(Dataset):
    def __init__(self, ordered_dataset, ratio, batch_size):

        self.ordered_dataset = ordered_dataset
        self.ratio = ratio
        self.batch_size = batch_size
        self.count = 0
        
    def __len__(self):
        return len(self.ordered_dataset)


    def __getitem__(self, index):
        self.count += 1

        if self.batch_size == 1:
            same_class = np.random.choice([1, 0], 1, p = [self.ratio, 1 - self.ratio])[0]
        else:
            same_class = (self.count / self.batch_size) < self.ratio

        if self.count == self.batch_size:
            self.count = 0

        if same_class:
            im1, im2 = self.ordered_dataset.get_same_class()
            
        else: 
            im1, im2 = self.ordered_dataset.get_diff_class()
        
        return im1, im2, same_class


class dataset_cat_no_cat(Dataset):
    def __init__(self, cat_images, background_images):
        self.cat_images = cat_images
        self.background_images = background_images

        self.output_which_image = 0  # 0 = background, 1 = cat

        self.n_class = len(self.cat_images)
        self.img_per_class = len(self.cat_images[0])

        self.length = self.n_class*self.img_per_class  # on assume que les longueurs de cat_images et background_images sont pareils

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i_class = random.randint(0, self.n_class - 1)
        i_img = random.randint(0, self.img_per_class - 1)

        selected_class = self.cat_images[i_class] if self.output_which_image else self.background_images[i_class]
        selected_image = selected_class[i_img]

        label = self.output_which_image

        self.output_which_image = 1 - self.output_which_image

        return selected_image, label
    






