import os
import numpy as np
import matplotlib.pyplot as plt

import glob

import torch

import torch.nn as nn
import torch.optim as optim

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from torch.utils.data import DataLoader





from MyDataLoader import ShuffleDataLoader
# from FileReader import get_picture_tensors
from FileReaderOther import get_picture_tensors
# from MyModels import SimpleCNN, CatNet
from ModelEvaluation import eval_model

from CatNet import CatNet


# import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, catnet):
        super(SiameseNetwork, self).__init__()
        self.catnet = catnet

    def forward_one(self, x):
        return self.catnet(x)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2




nbIt = 1
for i in range(nbIt):
    # n_classes = 64  # multiple de 2, sa aide tu?
    n_classes = 32
    # n_classes = 4
    # root_dir = "normal_prep_datasets/dataset_chat_downscale"
    root_dir = "normal_prep_datasets/dataset_chat_downscale_no_background"
    directory_savefile = os.path.join('cat_features', 'autoencodeur', '1er')
    required_train_imgs = 10
    required_test_imgs = 1
    
    ylim = None
    # ylim = 0, 5




    # À commenter et décommenter au besoin
    # # 1) Data loading
    print(f'Téléchargement des données...')

    (train_images, val_images, test_images, 
    train_labels, val_labels, test_labels,
    info) = get_picture_tensors(root_dir, 
                                n_classes = n_classes,
                                required_train_imgs = required_train_imgs,
                                required_test_imgs = required_test_imgs,
                                # shuffle_directories = True,
                                # shuffle_images = True,
                                shuffle_directories = False,
                                shuffle_images = False,
                                show_progress = False)

    n_classes = info['n_classes']
    label_to_directory = info['label_to_directory']

    print(f'Done!')


    # À commenter et décommenter au besoin
    # # 2) Extraction et sauvegarde des features
    print(f'Sauvegarde des features...')

    # model = CatNet(n_classes, output_layer = 'f')
    # model = CatNet(n_classes, output_layer = 'ff')
    
    # model = CatNet(n_classes, output_layer = 'ff')

    # # 1
    # classifieur = nn.Sequential(
    #     # nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(512, 128),
    #     nn.ReLU(inplace=True)
    # )

    classifieur = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        # nn.BatchNorm1d(128),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
    )

    # model = CatNet(cnn_backbone = 'resnet18', classifier = classifieur)
    # # model.load_parameters_from_file('resnet_512_to_128.pth') # avec relu
    # # model.load_parameters_from_file('resnet_512_to_128_minDiff.pth') # pas relu
    # # model.load_parameters_from_file('resnet_512_to_128_BATCH.pth') # avec relu
    
    # # model.load_parameters_from_file('resnet_512_to_128_BATCH_2layer.pth') # sans relu 2e
    # model.load_parameters_from_file('resnet_512_to_128_BATCH_2layer_minSame.pth') # avec relu 2e
    









    encoder = nn.Sequential(
        nn.Linear(9216, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 128),
    )

    # based_model = CatNet(cnn_backbone = 'alexnet', num_classes = 1024, classifier = encoder)
    model = CatNet(cnn_backbone = 'alexnet', classifier = encoder, append_classifier = False)
    # based_model = CatNet(cnn_backbone = 'alexnet', freeze_all_layer = False, classifier = classifieur)

    model.load_parameters_from_file('alexnet_autoencodeur_9216_1028_128_3run.pth')








    # # Load the saved parameters into the model
    # model_path = 'model_path.pth'
    # model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        for image, label in zip(train_images, train_labels):
            image = image.unsqueeze(0)
            features = model(image).numpy()

            label = label.numpy().argmax()  # [0 0 1 0 ...] -> 2
            filename = label_to_directory[label] + '.csv'
            filename = os.path.join(directory_savefile, filename)

            with open(filename, 'a') as f:
                np.savetxt(f, features.reshape(1, -1), fmt='%1.6f', delimiter=",")

    print('Done!')


    # # 3) Lecture des features
    print(f'Lecture des features du csv...')

    csv_files = glob.glob(os.path.join(directory_savefile, "*.csv"))

    # csv_files = [csv_files[0]]

    for csv_file in csv_files:
        filename = csv_file[:-4]

        cat_number = filename[-4:]

        with open(csv_file, 'r') as f:
            data = np.genfromtxt(f, delimiter=",")
        
        n_images, n_features = data.shape

        fig, ax = plt.subplots(1, 1, figsize=(15, 3))

        ax.set_xlabel('Features (-)')
        ax.set_ylabel('Value (-)')
        ax.set_title(f'Cat {cat_number}')
        ax.set_xlim(0, n_features)

        # ax.set_ylim(0, 50)
        # ax.set_ylim(-12, 12)
        # ax.set_ylim(0, 12)
        # ax.set_ylim(0, 5)
        
        if ylim is not None:
            ax.set_ylim(ylim)

        x_axis = np.arange(0, n_features, 1)

        mean_feature = 0*data[0, :]
        
        for i in range(n_images):
            feature = data[i, :]
            ax.scatter(x_axis, feature)
            mean_feature += feature



        mean_feature /= n_images
        # ax.scatter(x_axis, mean_feature)

        fig.tight_layout()
        # plt.show()
        
        png_file = filename + '.png'
        # filename = os.path.join(directory_savefile, filename)
        # print(png_file)
        # break
        fig.savefig(png_file, dpi = 300)

        print('Done!')













    # prochaine étape -> passer le réseau siamois de antoine là dessus
    # deux possibilité
    # une pour différencier les features qui distinguent un chat d'un autre
    # fec tu lui donnes une photo d'un chat, une photo d'un autre, et il doit
    # dire si ils sont pareils ou pas, tu entraines ton output pour avoir
    # les bons features... MAIS AUSSI, en parallèle, tu entraines sur
    # tu lui donnes des photos avec background et sans background, et tu
    # veux aller chercher dans les features les features qui ont rapport
    # avec le chat uniquement! sa pourrait aider à traiter des photos
    # peut-être...

    # hey est-ce qu'on pourrait savoir c'est quoi le cell qui a pris la photo
    # juste en passant une photo aléatoire dans un modèle? ahah
    
    # from sklearn.svm import SVC
    # # Train an SVM classifier
    # svm_classifier = SVC()
    # svm_classifier.fit(features_train, labels_train)

    # # Make predictions
    # predictions = svm_classifier.predict(features_test)

    # # Evaluate accuracy
    # accuracy = accuracy_score(labels_test, predictions)
    # print(f"Accuracy: {accuracy * 100:.2f}%")

    print('')