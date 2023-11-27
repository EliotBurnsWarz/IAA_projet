import torchvision.models as models
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super().__init__()

        if pretrained:
            self.model = models.resnet18(weights='DEFAULT', progress = False) 
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model = models.resnet18(weights=None, progress = False)
            
        dim_before_fc = self.model.fc.in_features

        # de base, requires_grad = True
        self.model.fc = nn.Linear(dim_before_fc, num_classes)

    def forward(self, x):
        x = self.model.forward(x)
        return x
