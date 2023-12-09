import torchvision.models as models
import torch.nn as nn


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
    
    def features(self, x):
        x = self.model.features(x)
        return x


class BiggerCatNet(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super().__init__()

        if pretrained:
            self.model = models.alexnet(weights = 'DEFAULT', progress = False) 
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model = models.alexnet(weights = None, progress = False)
            
            
        classification_layers = nn.Sequential(nn.Linear(9216, 100),
                                                nn.ReLU(inplace=True),
                                                nn.BatchNorm1d(100),
                                                nn.Linear(100, 100),
                                                nn.ReLU(inplace=True),
                                                nn.BatchNorm1d(100),
                                                nn.Linear(100, num_classes)
                                            )
        
        #classification_layers = nn.Sequential(nn.Linear(9216, num_classes))
        
        self.model.classifier = classification_layers
                 
    def forward(self, x):
        x = self.model.forward(x)
        return x


class FeatureExtractionCNN(nn.Module):
    def __init__(self, pretrained = True, train_hidden_layer = False):
        super().__init__()

        if pretrained:
            self.model = models.resnet18(weights = 'DEFAULT', progress = False) 

            if not train_hidden_layer:  # permet de loader les paramètres par défaut de resnet, mais de les optimiser quand même
                for param in self.model.parameters():
                    param.requires_grad = False
        else:
            self.model = models.resnet18(weights = None, progress = False)
            
        dim_before_fc = self.model.fc.in_features       
        self.model.fc = nn.Sequential(
            nn.Linear(dim_before_fc, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128))
                
    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, input1, input2):
        input1 = input1.unsqueeze(dim=0)
        input2 = input2.unsqueeze(dim=0)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    
    

if __name__ == '__main__':
    model = BiggerCatNet(5)
