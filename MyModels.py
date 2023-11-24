import torchvision.models as models
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super().__init__()
        self.model = models.resnet18(pretrained = pretrained, progress=False)   
        if pretrained:
            # on gèle les paramètres
            for param in self.model.parameters():
                param.requires_grad = False
        
        dim_before_fc = self.model.fc.in_features

        # de base, requires_grad = True
        self.model.fc = nn.Linear(dim_before_fc, num_classes) 
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.model.forward(x)
        return self.sigmoid(x)
