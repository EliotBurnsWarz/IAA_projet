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
fig_name = 'heatmap_acc_4x4'
# fig_extensions = ['png']  # test number_of_cats
fig_extensions = ['png', 'svg']  # pour saver



x = np.arange(0, 4)
y = np.arange(0, 4)

X, Y = np.meshgrid(x, y)
#train sur alexnet

Z = np.array([[95.3125, 78.125, 35.9375, 26.5625],  # train no centering
     [81.25, 93.75, 51.5625, 20.3125],  # downscale
     [39.0625, 73.4375, 92.1875, 10.9375],  # no background
     [35.9375, 35.9375, 6.25, 85.9375]])  # no cat

# Z = Z.transpose()

# # Figure
fig, ax = plt.subplots(1, 1, figsize=(3, 3))

ax.xaxis.tick_top()
ax.set_xlabel('Testé sur')
ax.xaxis.set_label_position('top')

plt.gca().invert_yaxis()

ax.set_ylabel('Entrainé sur')

mesh = ax.pcolormesh(X, Y, Z, cmap = 'Reds', rasterized=True)

for i in x:
    for j in y:
        # acc = np.round(Z[i, j], 1)
        acc = np.round(Z[j, i], 1)

        color = 'w' if acc > 30 else 'k'

        ax.text(i, j, acc, c = color, horizontalalignment = 'center', verticalalignment = 'center')


# 
# ax.set_xticks(np.arange(0, 61, 10))
# ax.set_yticks(np.arange(0, 61, 10))

# ax.plot(x_wrong_predictions, y_wrong_predictions, 'bs', markersize = 1.5)

# plt.grid(True)

fig.tight_layout()
ax.set_aspect('equal', adjustable='box')

for extension in fig_extensions:
    filename = os.path.join(saving_directory, 
        f"{fig_name}.{extension}") 
    
    fig.savefig(filename, dpi = 300)

plt.show()