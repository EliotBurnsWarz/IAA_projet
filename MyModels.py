import torchvision.models as models
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, pretrained = True, train_hidden_layer = False):
        super().__init__()

        if pretrained:
            self.model = models.resnet18(weights = 'DEFAULT', progress = False) 

            if not train_hidden_layer:  # permet de loader les paramètres par défaut de resnet, mais de les optimiser quand même
                for param in self.model.parameters():
                    param.requires_grad = False
        else:
            self.model = models.resnet18(weights = None, progress = False)
            
        dim_before_fc = self.model.fc.in_features

        # de base, requires_grad = True
        self.model.fc = nn.Linear(dim_before_fc, num_classes)

    def forward(self, x):
        x = self.model.forward(x)
        return x


class CatNet(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super().__init__()

        if pretrained:
            self.model = models.alexnet(weights = 'DEFAULT', progress = False) 
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model = models.alexnet(weights = None, progress = False)
               
        self.model.classifier = self.model.classifier[0:6].append(nn.Linear(4096, num_classes))


    def forward(self, x):
        x = self.model.forward(x)
        return x



if __name__ == '__main__':
    model = CatNet(5)
