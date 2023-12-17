
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin


    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        similarity = euclidean_distance

        # similarity = F.cosine_similarity(output1, output2)  # bof..
        
        target = target.int()

        loss_contrastive = torch.mean(target * torch.pow(similarity, 2) +
                                      (1 - target) * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))

        return loss_contrastive
    

class ContrastiveLoss_old(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin


    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        
        return loss


class PCALoss(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)


    def forward(self, input1, input2):
        data1 = input1.cpu().detach().numpy()
        data2 = input2.cpu().detach().numpy()

        data1 = data1.reshape(data1.shape[0], -1)
        data2 = data2.reshape(data2.shape[0], -1)

        transformed1 = self.pca.fit_transform(data1)
        transformed2 = self.pca.transform(data2)

        transformed1 = torch.tensor(transformed1, requires_grad=True, dtype=input1.dtype, device=input1.device)
        transformed2 = torch.tensor(transformed2, requires_grad=True, dtype=input2.dtype, device=input2.device)

        loss = torch.tensor(0.0, dtype=input1.dtype, device=input1.device)

        loss += torch.mean((transformed1 - transformed2)**2)

        return loss
