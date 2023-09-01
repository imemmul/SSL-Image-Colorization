from torch.utils.data import Dataset
import torch
import numpy as np
from dataset.visualize_dataset import visualize_map
import os
import hydra
from torchvision.transforms import Compose
from PIL import Image
import torchvision.transforms as transforms

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255


def create_cmap(classes:dict):
    return {k: list(np.random.choice(range(256), size=3)) for k,v in classes.items()}

def collate_fn(inputs):
    batch = dict()
    batch["images"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_images"] = [i[2] for i in inputs]
    batch["original_segmentation_maps"] = [i[3] for i in inputs]
    return batch

def check_dataset(dataset_dir):
    for dir in os.listdir(dataset_dir):
        img_dir = os.path.join(dataset_dir, dir)
        img = np.array(Image.open(img_dir))
        print(img.shape)

class CustomDataset(Dataset):
    def __init__(self, dataset, transform:Compose, classes:dict) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.classes = classes
        self.normalization = transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index): # dataset[0]
        item = self.dataset[index]
        original_image = torch.Tensor(np.array(Image.open(item['image'])))
        original_image_map = torch.LongTensor(np.array(Image.open(item['label'])))
        original_image = original_image.permute(2, 0, 1)
        image = self.transform(img=original_image)
        image = self.normalization(image)
        target = self.transform(img=original_image_map.unsqueeze(0))
        return image, target, original_image, original_image_map

def load_dataset_foodseg103(dataset_dir:str):
    image_sets_folder = os.path.join(dataset_dir, "ImageSets")
    category_id_file = os.path.join(dataset_dir, "category_id.txt")
    images_folder = os.path.join(dataset_dir, "Images")
    ann_dir = os.path.join(images_folder, "ann_dir")
    img_dir = os.path.join(images_folder, "img_dir")
    with open(os.path.join(image_sets_folder, "train.txt"), "r") as train_file:
        train_ids = [line.strip() for line in train_file]
    with open(os.path.join(image_sets_folder, "test.txt"), "r") as test_file:
        test_ids = [line.strip() for line in test_file]
    categories = {}
    with open(category_id_file, "r") as cat_file:
        for line in cat_file:
            category_id, category_name = line.split("\t")
            categories[category_id] = category_name
        print(f"Total {len(categories)} classes have been found.")
    train_dataset = []
    val_dataset = []
    for image_id in train_ids + test_ids:
        if image_id in train_ids:
            train_dataset.append({
                "image": os.path.join(img_dir, "train", image_id),
                "label": os.path.join(ann_dir, "train", image_id[:-3] + "png")
            })
        else:
            val_dataset.append({
                "image": os.path.join(img_dir, "test", image_id),
                "label": os.path.join(ann_dir, "test", image_id[:-3] + "png")
            })
    return train_dataset, val_dataset, categories
    