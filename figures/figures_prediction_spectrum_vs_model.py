import os
import glob
import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F



from MyDataLoader import OrderedDataSet, SiameseDataLoader, ShuffleDataLoader
from FileReader import get_picture_tensors
# from MyModels import FeatureExtractionCNN, CatNet
from ModelEvaluation import eval_model

from CatNet import CatNet















from scipy import interpolate
from scipy.optimize import curve_fit



from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc






saving_directory = 'figures'
fig_name = 'prediction_spectrum_vs_model'
# fig_extensions = ['png']  # test number_of_cats
fig_extensions = ['png', 'svg']  # pour saver


# root_directory = "normal_prep_datasets/dataset_chat_downscale/"
root_directory = "normal_prep_datasets/dataset_chat_downscale_no_background/"

n_classes = 32
required_train_imgs = 10
required_test_imgs = 1


(train_images, val_images, test_images, 
 train_labels, val_labels, test_labels, n_classes) = get_picture_tensors(root_directory=root_directory,
                                                              n_classes=n_classes, 
                                                              required_train_imgs=required_train_imgs, 
                                                              required_test_imgs=required_test_imgs,
                                                            #   use_selected_eval_datasets = True,
                                                              use_selected_eval_datasets = False)


classe_choisi = 24 #4

image = val_images[classe_choisi]
label = val_labels[classe_choisi]



param_alexnet = 'alexnet_32outputs_8epochs.pth'
param_mobilenet = 'mobilenet_v2_32outputs_8epochs.pth'
param_resnet = 'resnet18_32outputs_8epochs.pth'
param_vgg = 'vgg19_32outputs_8epochs.pth'

list_params = [param_alexnet, param_mobilenet, param_resnet, param_vgg]

models_name = ['alexnet', 'mobilenet_v2', 'resnet18', 'vgg19']


# # Figure
fig, axs = plt.subplots(len(list_params), 1, figsize=(3, 4), gridspec_kw = {'hspace':0})

x_axis = np.arange(0, n_classes)

color = [f'C{i}' for i in range(len(list_params))]

# ax.set_title('Truth : ' + str(classe_choisi) + '\n Predicted : ' + str(predicted.item()))

for i, (cnn_backbone, param, ax) in enumerate(zip(models_name, list_params, axs)):
    if i + 1 != len(list_params):
        # ax.set_xticklabels([])
        ax.tick_params(labelbottom=False)  
    
    ax.set_xlim(0 - 0.5, n_classes - 0.5)
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_yticks(np.arange(0, 1.1, 0.5))

    ax.set_ylabel('Probabilité')

    model = CatNet(cnn_backbone = cnn_backbone, num_classes = n_classes)
    model.load_parameters_from_file(param)

    model.eval()
    with torch.no_grad():

        outputs = model(image.unsqueeze(0))

        softmax_output = F.softmax(outputs).squeeze(0)

        _, predicted = torch.max(outputs.data, 1)

    # Probabilité
    ax.bar(x_axis, softmax_output, width = 1*n_classes/50, label = cnn_backbone, color = color[i])
    ax.bar(x_axis, label, width = 0.4*n_classes/50, color = 'black')
    # ax.bar(x_axis, softmax_output/max(softmax_output), width = 0.5*n_classes/50, label = 'Prediction', color = 'C1')
    ax.set_ylim(0, 1.3)

    ax.legend(loc = 'upper left')
    # ax.legend(loc = 'upper right')


axs[-1].set_xlabel('Classes')

fig.tight_layout()

for extension in fig_extensions:
    filename = os.path.join(saving_directory, 
        f"{fig_name}.{extension}") 
    
    fig.savefig(filename, dpi = 600)

plt.show()