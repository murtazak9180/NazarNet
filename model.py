import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

resnet = models.resnet50(pretrained=True)


class ResnetEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)  #The global average pooling layer already exist in resnet50 before the fc

    def forward(self, x):
        x = self.resnet(x)
        x = F.normalize(x, p=2, dim=1)   #L2 normalization. Makes all the points lie on a sphere, makes lerning more stable. 
        return x 
    
