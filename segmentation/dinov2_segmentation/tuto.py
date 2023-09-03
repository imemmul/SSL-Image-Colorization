import torch
from torch import nn

m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
print(input.shape)
print(target.shape)
output = loss(m(input), target)
output.backward()