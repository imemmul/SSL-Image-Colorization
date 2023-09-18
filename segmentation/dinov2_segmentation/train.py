from dataset.dataset import CustomDataset, load_dataset_foodseg103, create_cmap, check_dataset, collate_fn, load_dataset_fishency
from dataset.visualize_dataset import visualize_map
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.data import DistributedSampler, DataLoader
from torchvision.transforms import transforms
import hydra
from omegaconf import DictConfig, OmegaConf
from models.dino_seg import Dinov2ForSemanticSegmentation, ABS_PATH
from tqdm.auto import tqdm
from models.utils import compute_iou, compute_mean_accuracy
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.distributed as dist
import os
from memory_profiler import profile
import yaml

os.environ['TORCH_USE_CUDA_DSA'] = '1'


device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize(size=(448, 448), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
])
val_transform = transforms.Compose([
    transforms.Resize(size=(448, 448), antialias=True),
])


def validate_model(model, val_dataloader, cfg, device, classes):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    total_macc = 0.0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            imgs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(imgs, labels=labels, output_attentions=True)
            loss = outputs["loss"]

            predicted = outputs["logits"]
            predicted_mask = (predicted > cfg['threshold']).int()
            miou = compute_iou(predicted_mask, labels)
            macc = compute_mean_accuracy(predicted_mask, labels, len(classes))

            total_loss += loss.mean().item()
            total_miou += miou
            total_macc += macc

    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    avg_macc = total_macc / num_batches

    print("Validation Loss:", avg_loss)
    print("Validation Mean IoU:", avg_miou)
    print("Validation Mean Accuracy:", avg_macc)

    model.train()
    return avg_loss

def train_model(model:DataParallel, cfg, train_dataloader, val_dataloader, classes):
    epochs = int(cfg['epochs'])
    train_losses = []
    val_losses = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']))

    model.to(device)

    model.train()

    for epoch in range(epochs):
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            imgs = batch["images"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(imgs, labels=labels, output_attentions=True)
            loss = outputs["loss"]
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                predicted = outputs["logits"]
                predicted_mask = (predicted > cfg['threshold']).int() # extra tensor
                predicted_process = predicted_mask[0].detach().cpu().numpy().squeeze()
                plt.imsave(f"{ABS_PATH}processing_predicted.png", predicted_process, cmap="binary")
        miou = compute_iou(predicted_mask, labels)
        macc = compute_mean_accuracy(predicted_mask, labels, len(classes))
        print("Train Loss:", loss.mean().item())
        print("Train mIoU:", miou)
        print("Train mAcc:", macc)
        print("--------------------------------------------------")
        val_loss = validate_model(val_dataloader=val_dataloader,cfg=cfg, model=model, device=device, classes=classes)
        val_losses.append(val_loss)
        train_losses.append(loss.mean().item())
    model.module.save_model_checkpoints()
    plt.figure(figsize=(10, 5))  # Optional: Set the figure size
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', linestyle='-')

    # Add labels and a legend
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(f'{ABS_PATH}loss_plot.png')

#@hydra.main(config_path="./configs", config_name="train_config.yaml") # if hydra version >= 1.3.2 include version_base=None, hydra consuming too much memory
def main():
    config_dir = "./configs/train_config.yaml"
    with open(config_dir, "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    # Now, config_dict contains the configuration from the YAML file as a dictionary
    print(cfg)

    if cfg['dataset_name'] == "FoodSeg103":
        train_data, val_data, classes = load_dataset_foodseg103(dataset_dir=cfg['dataset_dir'])
    elif cfg['dataset_name'] == "Fishency":
        train_data, val_data, classes = load_dataset_fishency(dataset_dir=cfg['dataset_dir'])
        print(classes)
        print(f"Total train samples: {len(train_data)}, val samples: {len(val_data)} with {len(classes)} classes")
    train_dataset = CustomDataset(dataset=train_data, transform=train_transform, classes=classes, binary_seg=True) # bak simdi bunu boyle yazdim ya
    test_dataset = CustomDataset(dataset=val_data, transform=val_transform, classes=classes, binary_seg=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], collate_fn=collate_fn)
    val_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], collate_fn=collate_fn)
    model = Dinov2ForSemanticSegmentation(cfg, len(classes)).to("cuda")
    fine_tune_ratio = 0.8
    total_params = len(list(model.parameters()))
    fine_tune_index = int(total_params * fine_tune_ratio)
    print(f"fine_tune_index: {fine_tune_index}")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"training {cfg['model_name']} with {total_trainable_params} parameters before fine-tuning")
    for name, param in list(model.named_parameters())[:fine_tune_index]:
            param.requires_grad = False
    
    # ========================================================================
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"training {cfg['model_name']} with {total_trainable_params} parameters")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = DataParallel(model, device_ids=[0, 1])
    # ========================================================================
    train_model(model=model, cfg=cfg, train_dataloader=train_dataloader, val_dataloader=val_dataloader, classes=classes)


if __name__ == "__main__":
    main()