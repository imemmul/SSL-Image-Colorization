import torch
from torch import nn

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, W=32, H=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = W
        self.height = H
        self.bn = torch.nn.SyncBatchNorm(in_channels)
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))
    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)
        embeddings = self.bn(embeddings)
        return self.classifier(embeddings)
    
        