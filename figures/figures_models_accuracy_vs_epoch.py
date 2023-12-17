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



from matplotlib import rc






saving_directory = 'figures'
fig_name = 'models_accuracy_vs_epoch'
# fig_extensions = ['png']  # test number_of_cats
fig_extensions = ['png', 'svg']  # pour saver

filename_alexnet = 'alexnet_32outputs_1e-4_batch8_epoch16_train10_val_1.csv'
filename_mobilenet = 'mobilenetv2_32outputs_1e-4_batch8_epoch16_train10_val_1.csv'
filename_resnet = 'resnet18_32outputs_1e-4_batch8_epoch16_train10_val_1.csv'
filename_vgg = 'vgg19_32outputs_1e-4_batch8_epoch16_train10_val_1.csv'

filenames = [filename_alexnet, filename_mobilenet, filename_resnet, filename_vgg]
labels = ['alexnet', 'mobilenet_v2', 'resnet18', 'vgg19']





fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.set_xlabel('Époch')
ax.set_ylabel('Précision (%)')

color = ['C0', 'C1', 'C2', 'C3']

for i, (filename, label) in enumerate(zip(filenames, labels)):
    with open(filename, 'r') as f:
        data = np.genfromtxt(f, delimiter=",")
    
    acc_train = data[0, :]
    acc_val = data[1, :]

    ax.plot(acc_train, '-', color = color[i], label = label)
    ax.plot(acc_val, '--', color = color[i])
    



    # with torch.no_grad():
    # # images = val_images[i]
    # # labels = val_labels[i]
    # images = test_images[i]
    # labels = test_labels[i]

    # images = images.unsqueeze(0)
    # outputs = model(images)

    # softmax_output = F.softmax(outputs, dim = 1).squeeze(0)

    # _, predicted = torch.max(outputs.data, 1)

    # if i != predicted:
    #     x_wrong_predictions.append(predicted)
    #     y_wrong_predictions.append(i)

    # Z[i, :] = softmax_output.detach().numpy()


# # # Figure
# fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# ax.xaxis.tick_top()
# ax.set_xlabel('Prédiction du modèle')
# ax.xaxis.set_label_position('top')

# plt.gca().invert_yaxis()

# ax.set_ylabel('Classe attendue')

# mesh = ax.pcolormesh(X, Y, Z, cmap = 'Reds', rasterized=True)

# # 
# ax.set_xticks(np.arange(0, 61, 10))
# ax.set_yticks(np.arange(0, 61, 10))














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




# ax.plot(x_wrong_predictions, y_wrong_predictions, 'bs', markersize = 1.5)

# plt.grid(True)
    
ax.legend(loc = 'best')

fig.tight_layout()

for extension in fig_extensions:
    filename = os.path.join(saving_directory, 
        f"{fig_name}.{extension}") 
    
    fig.savefig(filename, dpi = 300)

plt.show()