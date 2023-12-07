import os
import glob

import numpy as np

from PIL import Image, ImageFilter, ImageChops

import torch

import torchvision.transforms.functional as F

import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50

from torchvision.ops import masks_to_boxes

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = 'tight'

import time


from dataset_preprocessing import preprocess_original_image


def keep_same_elements(list1, list2):
    same_elements = []

    for element_of_list1 in list1:
        for element_of_list2 in list2:
            if element_of_list1 == element_of_list2:
                same_elements.append(element_of_list1)

    return same_elements


def keep_same_filenames(list1, list2):
    same_elements = []

    for element_of_list1 in list1:
        _, filename_1 = os.path.split(element_of_list1)
    
        for element_of_list2 in list2:
            _, filename_2 = os.path.split(element_of_list2)

            if filename_1 == filename_2:
                    same_elements.append(element_of_list1)

    return same_elements


def info(variable, how_many_to_show = 8):
    print(f'Type : {type(variable)}')

    print(f'Len : {len(variable)}')

    if len(variable) > how_many_to_show:
        part_1 = variable[0:int(how_many_to_show/2)]
        part_2 = variable[-int(how_many_to_show/2):]
        print(f'Contenu : {str(part_1)[:-1]}, ..., {str(part_2)[1:]}')
    else:
         print(f'Contenu : {variable}')

    print('')


def retrieve_folder(path):
    folder = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return folder
    




if __name__ == '__main__':

    directory_cat_background = 'dataset_chat_downscale'
    directory_cat = 'dataset_chat_downscale_no_background'
    directory_background = 'dataset_no_cat'

    subdirectories_cat_background = retrieve_folder(directory_cat_background)
    subdirectories_cat = retrieve_folder(directory_cat_background)
    
    subdirectories = c = keep_same_elements(subdirectories_cat_background,
                                            subdirectories_cat)

    # info(subdirectories)

    # subdirectories = ['0105']


    for subdirectory in subdirectories:
        subdirectory_cat_background = os.path.join(directory_cat_background, subdirectory)
        subdirectory_cat = os.path.join(directory_cat, subdirectory)
        subdirectory_background = os.path.join(directory_background, subdirectory)

        jpg_files_cat_background = glob.glob(os.path.join(subdirectory_cat_background, "*.jpg"))
        jpg_files_cat = glob.glob(os.path.join(subdirectory_cat, "*.jpg"))

        jpg_files_cat_background = keep_same_filenames(jpg_files_cat_background, 
                                                       jpg_files_cat)

        # jpg_files = jpg_files[-2:]  # si on veut sélectionner un range
        # jpg_files = [jpg_files[3]]  # si on veut sélectionner un range

        print(str(len(jpg_files_cat)) + ' files in the subdirectory ' + subdirectory)

        # On crée le/les dossiers s'ils n'existent pas
        try:
            # os.makedirs(os.path.abspath(os.getcwd()) + '\\' + file_directory_with_background.replace("/", "\\"))  # créer le dossier
            os.makedirs(os.path.join(os.path.abspath(os.getcwd()), subdirectory_background))  # créer le dossier
        except (OSError, FileExistsError) as error: 
            pass

        for jpg_file_cat_background in jpg_files_cat_background:
            _, filename = os.path.split(jpg_file_cat_background)

            jpg_file_cat = os.path.join(subdirectory_cat, filename)
            jpg_file_background = os.path.join(subdirectory_background, filename)

            with Image.open(jpg_file_cat_background) as im_cat_background:
                with Image.open(jpg_file_cat) as im_cat:
                    im_background = ImageChops.subtract(im_cat_background, im_cat)

                    im_background.save(jpg_file_background)

            # print(filename + ' -> Done')  # trop rapide...

        print('Done!')
        print(' ')
