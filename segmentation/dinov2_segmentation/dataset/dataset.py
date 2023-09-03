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
    def __init__(self, dataset, transform:Compose, classes:dict, binary_seg) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.classes = classes
        self.binary_seg = binary_seg
        self.normalization = transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index): # dataset[0]
        item = self.dataset[index]
        if self.binary_seg:
            original_image = torch.Tensor(np.array(Image.open(item['image'])))
            original_image_map = torch.LongTensor(np.array(Image.open(item['label']).convert("L")))
        original_image = original_image.permute(2, 0, 1)
        image = self.transform(img=original_image)
        image = self.normalization(image)
        target = self.transform(img=original_image_map.unsqueeze(0))
        # pixel values consist of not only 255 or 0, so need to scale all of them into 0 and 1,
        target = (target != 0).float()
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

def load_dataset_fishency(dataset_dir:str):
    classes = {0: "background",
              1: "fish"}
    train_dataset = []
    val_dataset = []
    for split in os.listdir(dataset_dir):
        split_dir = os.path.join(dataset_dir, split)
        for cat in os.listdir(split_dir):
            if cat == "fgr":
                images_dir = os.path.join(split_dir, cat)
            else:
                labels_dir = os.path.join(split_dir, cat)
        for img_folder in sorted(os.listdir(images_dir)):
            img_folder_dir = os.path.join(images_dir, img_folder)
            for img in sorted(os.listdir(img_folder_dir)):
                if split == "train":
                    train_dataset.append({
                        "image": os.path.join(img_folder_dir, img),
                        "label": f"{os.path.join(labels_dir, img_folder)}/{img}"
                    })
                else:
                    val_dataset.append({
                        "image": os.path.join(img_folder_dir, img),
                        "label": f"{os.path.join(labels_dir, img_folder)}/{img}"
                    })

    return train_dataset, val_dataset, classes