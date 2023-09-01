import torch
from torch import nn
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.builder import HEADS
from mmseg.models.utils import resize

@HEADS.register_module()
class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, W=32, H=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = W
        self.height = H
        self.bn = torch.nn.SyncBatchNorm(in_channels)
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))
    def forward(self, embeddings):
        embeddings = self.bn(embeddings)
        return self.classifier(embeddings)

@HEADS.register_module()
class BN_Head(BaseDecodeHead):
    """
    They are only applying BN
    """
    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
    def _forward_features(self, x):
        """
        first forward features through transforms before classifying to match input size
        """
        x = self._transform_inputs(x)
        features = self.bn(x)
    
        