import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

label_blank = "/home/doga/Projects/emir_workspace/data/matting-data/FishencyVideo/train/pha/0027/00013.jpg"
label_fish = "/home/doga/Projects/emir_workspace/data/matting-data/FishencyVideo/train/pha/0027/00009.jpg"

train_transform = transforms.Compose([
    transforms.Resize(size=(448, 448), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
])

blank_target = torch.LongTensor(np.array(Image.open(label_blank).convert('L')))
fish_target = torch.LongTensor(np.array(Image.open(label_fish).convert('L')))
print(fish_target)
blank_target = (blank_target != 0)
fish_target = (fish_target != 0)
blank_target = train_transform(img=blank_target.unsqueeze(0)).squeeze().float()
fish_target = train_transform(img=fish_target.unsqueeze(0)).squeeze().float()
print(f"fish_target: {fish_target.shape}")
print(f"blank_target: {blank_target.shape}")
print(fish_target)
print(fish_target.cpu().numpy())
print(blank_target.cpu().numpy())
plt.imsave("./blank_target.png", blank_target.cpu().numpy(), cmap="binary")
plt.imsave("./fish_target.png", fish_target.cpu().numpy(), cmap="binary")