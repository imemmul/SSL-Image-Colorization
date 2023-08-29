from torch.utils.data import Dataset
import torch
import numpy as np
import os
import hydra
from torchvision.transforms import Compose
from PIL import Image
def create_cmap(classes:dict):
    return {k: list(np.random.choice(range(256), size=3)) for k,v in classes.items()}

class CustomDataset(Dataset):
    def __init__(self, dataset, transform:Compose, classes:dict) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.classes = classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        original_image = np.array(Image.open(item['image']))
        original_image_map = np.array(Image.open(item['label']))
        print(original_image.shape)
        print(original_image_map.shape)
        image = self.transform(img=original_image)
        target = self.transform(img=original_image_map)
        image = image.permute(2, 0, 1)
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
                "label": os.path.join(ann_dir, "test", image_id[:-3] + ".png")
            })
    return train_dataset, val_dataset, categories
    