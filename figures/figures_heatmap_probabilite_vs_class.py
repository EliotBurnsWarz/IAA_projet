import os
import glob
import numpy as np
import random

import matplotlib.pyplot as plt

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
fig_name = 'heatmap_probabilite_vs_class_train_background_test_background'
# fig_extensions = ['png']  # test number_of_cats
fig_extensions = ['png', 'svg']  # pour saver


# root_directory = "normal_prep_datasets/dataset_chat_no_centering/"
root_directory = "normal_prep_datasets/dataset_chat_downscale/"
# root_directory = "normal_prep_datasets/dataset_chat_downscale_no_background/"
# root_directory = "normal_prep_datasets/dataset_chat_downscale_no_cat/"

n_classes = 64
required_train_imgs = 10
required_test_imgs = 1


(train_images, val_images, test_images, 
 train_labels, val_labels, test_labels, n_classes) = get_picture_tensors(root_directory=root_directory,
                                                              n_classes=n_classes, 
                                                              required_train_imgs=required_train_imgs, 
                                                              required_test_imgs=required_test_imgs,
                                                            #   use_selected_eval_datasets = True,
                                                              use_selected_eval_datasets = False)


train_dataset = ShuffleDataLoader(train_images, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle=True)
val_dataset = ShuffleDataLoader(val_images, val_labels)

model = CatNet(cnn_backbone = 'mobilenet_v2', num_classes = n_classes)
# model.load_parameters_from_file('mobilenet_v2_64outputs_epoch4_.pth')  # train no background
model.load_parameters_from_file('mobilenet_v2_64outputs_epoch4_2.pth')  # train avec background





# # section code pour afficher la probabilité en sortie
# index = 8

# index 21 = 0023
# index 21 = 0026

# model.eval()

# with torch.no_grad():
#     images = val_images[index]
#     labels = val_labels[index]

#     images = images.unsqueeze(0)
#     outputs = model(images)

#     sig_output = F.sigmoid(outputs).squeeze(0)
#     softmax_output = F.softmax(outputs).squeeze(0)

#     _, predicted = torch.max(outputs.data, 1)


# from matplotlib.ticker import MultipleLocator

# x_axis = np.arange(0, n_classes, 1)

# fig, ax = plt.subplots(1, 1, figsize=(5, 3))

# ax.set_xlabel('Classe')
# ax.set_ylabel('Probabilité (-)')
# ax.set_xlim(0 - 0.5, n_classes - 0.5)
# ax.xaxis.set_minor_locator(MultipleLocator(1))
# ax.set_title('Truth : ' + str(index) + '\n Predicted : ' + str(predicted.item()))

# # Probabilité
# ax.bar(x_axis, labels, width = 1*n_classes/50, label = 'Truth')
# # ax.bar(x_axis, sig_output/max(sig_output), width = 0.3*n_classes/50, label = 'Prediction (norm)', color = 'C2')
# # ax.bar(x_axis, sig_output, width = 0.5*n_classes/50, label = 'Prediction', color = 'C1')
# # ax.bar(x_axis, sig_output/max(sig_output), width = 0.3*n_classes/50, label = 'Prediction (norm)', color = 'C2')
# ax.bar(x_axis, softmax_output, width = 0.5*n_classes/50, label = 'Prediction', color = 'C1')
# ax.set_ylim(0, 1)

# fig.tight_layout()

# plt.show()


# print('    Classe : ' + str(index))
# print('Prédiction : ' + str(predicted.item()))





model.eval()

x = np.arange(0, n_classes)
y = np.arange(0, n_classes)

X, Y = np.meshgrid(x, y)
Z = np.zeros((len(y), len(x)))

x_wrong_predictions = []
y_wrong_predictions = []

for i in range(n_classes):
    with torch.no_grad():
        # images = val_images[i]
        # labels = val_labels[i]
        images = test_images[i]
        labels = test_labels[i]

        images = images.unsqueeze(0)
        outputs = model(images)

        softmax_output = F.softmax(outputs, dim = 1).squeeze(0)

        _, predicted = torch.max(outputs.data, 1)

        if i != predicted:
            x_wrong_predictions.append(predicted)
            y_wrong_predictions.append(i)

    Z[i, :] = softmax_output.detach().numpy()


# # Figure
fig, ax = plt.subplots(1, 1, figsize=(3, 3))

ax.xaxis.tick_top()
ax.set_xlabel('Prédiction du modèle')
ax.xaxis.set_label_position('top')

plt.gca().invert_yaxis()

ax.set_ylabel('Classe attendue')

mesh = ax.pcolormesh(X, Y, Z, cmap = 'Reds', rasterized=True)

# 
ax.set_xticks(np.arange(0, 61, 10))
ax.set_yticks(np.arange(0, 61, 10))

ax.plot(x_wrong_predictions, y_wrong_predictions, 'bs', markersize = 1.5)

plt.grid(True)

fig.tight_layout()
ax.set_aspect('equal', adjustable='box')

for extension in fig_extensions:
    filename = os.path.join(saving_directory, 
        f"{fig_name}.{extension}") 
    
    fig.savefig(filename, dpi = 300)

plt.show()