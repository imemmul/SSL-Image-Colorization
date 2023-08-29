import collections
import os
from os.path import join
import io

import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import wget
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchmetrics import Metric
from torchvision import models
from torchvision import transforms as T

def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize])
# TODO add target normalization values into configs
# normalized values of trained dataset should be changed 
normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class ToTargetTensor(object):
    """
    simple used line transform to torch.Tensor
    """
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)