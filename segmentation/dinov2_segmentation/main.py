from torch.utils.data import Dataset
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataset, transform) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        item = self.dataset[index]
        original_image = np.array(item['image'])
        original_image_map = np.array(item['label'])
        transformed = self.transform(image=original_image, mask=original_image_map)
        image, target = torch.tensor(transformed['image']), torch.tensor(transformed['mask'])
        iamge = image.permute(2, 0, 1)
        return image, target, original_image, original_image_map