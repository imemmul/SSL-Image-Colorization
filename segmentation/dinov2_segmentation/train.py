from dataset.dataset import CustomDataset, load_dataset_foodseg103, create_cmap
from dataset.visualize_dataset import visualize_map
import torch
from torchvision.transforms import transforms
import hydra
from omegaconf import DictConfig, OmegaConf


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# train_data = CustomDataset(dataset=train_dataset, transform=train_transform)
# val_data = CustomDataset(dataset=val_dataset, transform=val_transform)

@hydra.main(config_path="./configs", config_name="train_config.yaml", version_base=None)
def main(cfg:DictConfig):
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    train_data, val_data, classes = load_dataset_foodseg103(dataset_dir=cfg.dataset_dir)
    train_dataset = CustomDataset(dataset=train_data, transform=train_transform, classes=classes)
    val_dataset = CustomDataset(dataset=val_data, transform=val_transform, classes=classes)
    colors = create_cmap(classes=classes)
    image, target, original_image, original_image_map = train_dataset[0]
    visualize_map(original_image, segmentation_map=original_image_map, colors=colors)

if __name__ == "__main__":
    main()