import torchvision.models as models
import torch.nn as nn

from torchsummary import summary

import numpy as np


import torch
import torch.nn as nn
import torchvision.models as models


class CatNet(nn.Module):
    """
    cnn_backbone : 'alexnet', 'resnet18', 'vgg19' ou 'mobilenet_v2' pour l'instant

    """
    def __init__(self, cnn_backbone = 'alexnet', pretrained = True, freeze_all_layer = True, 
                 num_classes = None, classifier = None, append_classifier = True):
        super().__init__()

        self.cnn_backbone = cnn_backbone.lower()

        # load the cnn and the classifier as two different entities
        self.cnn, self.classifier = self._create_cnn_and_classifier(self.cnn_backbone, pretrained)

        if freeze_all_layer:
            for param in self.parameters():
                param.requires_grad = False

        self._handle_classifier_option(num_classes, classifier, append_classifier)


    def _create_cnn_and_classifier(self, cnn_name, pretrained):
        weights = None

        if pretrained:
            weights = 'DEFAULT'

        # AlexNet
        if cnn_name == 'alexnet':
            alexnet = models.alexnet(weights = weights, progress = False)

            cnn_part = alexnet.features.append(alexnet.avgpool)
            classifier_part = alexnet.classifier

        # ResNet18
        elif cnn_name == 'resnet18':
            resnet18  = models.resnet18(weights = weights, progress = False)

            cnn_part = nn.Sequential(
                resnet18.conv1,
                resnet18.bn1,
                resnet18.relu,
                resnet18.maxpool,
                resnet18.layer1,
                resnet18.layer2,
                resnet18.layer3,
                resnet18.layer4,
                resnet18.avgpool
            )
            classifier_part = nn.Sequential(resnet18.fc)

        # VGG19
        elif cnn_name == 'vgg19':
            vgg19 = models.vgg19(weights = weights, progress = False)

            cnn_part = vgg19.features.append(vgg19.avgpool)
            classifier_part = vgg19.classifier

        # MobileNet V2
        elif cnn_name == 'mobilenet_v2':
            mobilenet_v2 = models.mobilenet_v2(weights = weights, progress = False)

            cnn_part = mobilenet_v2.features.append(nn.AdaptiveAvgPool2d((1, 1)))
            classifier_part = mobilenet_v2.classifier

        # assez facile d'ajouter beaucoup d'autres modèles avec d'autres elif

        else:
            raise ValueError("Unsupported CNN backbone")

        return cnn_part, classifier_part
    

    def _handle_classifier_option(self, num_classes, classifier, append_classifier):
        if num_classes is not None:
            dim_input_last_layer = self.classifier[-1].in_features
            self.classifier[-1] = nn.Linear(dim_input_last_layer, num_classes)

            append_classifier = True

        if isinstance(classifier, nn.Sequential):
            if append_classifier:
                self.classifier.append(classifier)
            else:
                self.classifier = classifier


    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

    def features(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)  # 9216 pour alexnet
        return x
    
        #                                                                                         # TODO
    # méthode qui permet de locker et délocker le training des paramètres de certaines couches
    # sa pourrait être utile lors de l'entrainement 
    # for param in self.classifier.parameters():
    #     # permettent d'activer et désactiver des couches, 
    #     param.requires_grad = False
    

    # def features(self, x):
    #     x = self.model.features(x)
    #     return self.cnn(x)


    def summary(self, input_size = (3, 224, 224)):
        summary(self, input_size = input_size, device = 'cpu')

    
    def get_trainable_layer_names(self, model_part = 'classifier'):
        all_layer_names = [name for name, _ in self.named_parameters()]

        def keep_name_only(old_list):
            # convert ['cnn.0.weight', 'cnn.0.bias', 'cnn.2.weight', 'cnn.2.bias', ...]
            # to ['cnn.0', 'cnn.2', ...]
            new_list = [name.rsplit('.', 1)[0] for name in old_list]
            new_list = list(dict.fromkeys(new_list))
            
            return new_list

        all_layer_names = keep_name_only(all_layer_names)

        if model_part == 'all':
            return all_layer_names
        
        elif model_part == 'cnn':
            return [name for name in all_layer_names if name.startswith('cnn')]
        
        elif model_part == 'classifier':
            return [name for name in all_layer_names if name.startswith('classifier')]
        
        else:
            raise ValueError("Invalid value for 'model_part'. Choose 'all', 'cnn', or 'classifier'.")


    #                                                                                         # TODO
    # faire deux méthodes plus générales qui permettent de sauvegarder
    # et loader les paramètres relative à une/quelques couches
    # parce que présentement, on save les poids du réseau au complet chaque fois...
    def save_parameters_to_file(self, filepath):
        torch.save(self.state_dict(), filepath)


    def load_parameters_from_file(self, filepath):
        self.load_state_dict(torch.load(filepath))


# permet de conserver la classe Catnet avec un output
class SiameseNetwork(nn.Module):
    def __init__(self, catnet, outputFunction = None):
        super().__init__()
        self.catnet = catnet

        self.outputFunction = nn.Identity()

        if outputFunction == 'relu':
            self.outputFunction = nn.ReLU(inplace=True)

    def forward_one(self, x):
        return self.catnet(x)


    def forward(self, input1, input2, label = None):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        if label is not None:
            if label:
                output1 = self.outputFunction(output1)
                output2 = self.outputFunction(output2)

        return output1, output2
    






# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, catnet, decoder):
        super().__init__()
        self.encoder = catnet  # Use CatNet as the encoder

        self.features = self.encoder.features

        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




if __name__ == '__main__':

    # classifieur = nn.Sequential(
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(512, 128),
    #     nn.ReLU(inplace=True)
    # )

    # classifieur = nn.Sequential(
    #     CustomPCALayer(output_dim = 128)  # initial = 9216
    #     # nn.Dropout(p=0.5),
    #     # nn.Linear(9216, 4096),
    #     # nn.ReLU(inplace=True),
    #     # nn.Dropout(p=0.5),
    #     # nn.Linear(4096, 1024),
    #     # nn.ReLU(inplace=True),
    #     # nn.Linear(1024, 128),
    # )



    # model = CatNet(cnn_backbone = 'alexnet')
    # # model = CatNet(cnn_backbone = 'resnet18')
    # model = CatNet(cnn_backbone = 'vgg19')

    # model = CatNet(cnn_backbone = 'alexnet', classifier = classifieur)
    # model = CatNet(cnn_backbone = 'resnet18', classifier = classifieur)
    # model.summary()
    
    # allName = model.get_trainable_layer_names()  # default : model_part = 'classifier'
    # # # allName = model.get_trainable_layer_names(model_part = 'all')
    # # # allName = model.get_trainable_layer_names(model_part = 'cnn')
    # print(allName)


    classifieur = nn.Sequential(
        nn.Linear(100, 5),
        nn.ReLU(inplace=True)
    )

    # si on donne un nombre de classe ET un classifieur, il prend le classifieur par
    # défaut, le modifie pour avoir n_class en sortie, puis append le classifieur qu'on
    # lui donne en argument.

    model = CatNet(cnn_backbone = 'alexnet', num_classes = 100, classifier = classifieur)
    # model = CatNet(cnn_backbone = 'resnet18', classifier = classifieur)
    model.summary()
    
    # allName = model.get_trainable_layer_names()  # default : model_part = 'classifier'
    # # # allName = model.get_trainable_layer_names(model_part = 'all')
    # # # allName = model.get_trainable_layer_names(model_part = 'cnn')
    # print(allName)
