from dataset.dataset import CustomDataset, load_dataset_foodseg103, create_cmap, check_dataset, collate_fn, load_dataset_fishency
from dataset.visualize_dataset import visualize_map
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.data import DistributedSampler, DataLoader
from torchvision.transforms import transforms
import hydra
from omegaconf import DictConfig, OmegaConf
from models.dino_seg import Dinov2ForSemanticSegmentation
from tqdm.auto import tqdm
from models.utils import compute_iou, compute_mean_accuracy
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.distributed as dist



device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize(size=(896, 896), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
])
val_transform = transforms.Compose([
    transforms.Resize(size=(896, 896), antialias=True),
])

def validate_model(model, val_dataloader, device, classes):
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

            predicted = outputs["logits"].argmax(dim=1)
            miou = compute_iou(predicted, labels)
            macc = compute_mean_accuracy(predicted, labels, len(classes))

            total_loss += loss.item()
            total_miou += miou
            total_macc += macc

    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    avg_macc = total_macc / num_batches

    print("Validation Loss:", avg_loss)
    print("Validation Mean IoU:", avg_miou)
    print("Validation Mean Accuracy:", avg_macc)

    model.train()

def train_model(model, cfg, train_dataloader, val_dataloader, classes):
    epochs = cfg.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    model.to(device)

    model.train()

    for epoch in range(epochs):
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            imgs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(imgs, labels=labels, output_attentions=True)
            loss = outputs["loss"]

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            with torch.no_grad():
                predicted = outputs["logits"].argmax(dim=1)
                predicted_process = predicted[0].detach().cpu().numpy().squeeze()
                plt.imsave("./processing_predicted.png", predicted_process)
                # print(f"type of predictions: {predicted.detach().cpu().numpy().shape}")
                # print(f"type of labels{labels.detach().cpu().numpy().shape}")
        torch.cuda.empty_cache()
        miou = compute_iou(predicted, labels)
        macc = compute_mean_accuracy(predicted, labels, len(classes))
        print("Train Loss:", loss.item())
        print("Train mIoU:", miou)
        print("Train mAcc:", macc)
        print("--------------------------------------------------")
        validate_model(val_dataloader=val_dataloader, model=model, device=device, classes=classes)
    torch.save(model.dinov2.state_dict(), "./backbone_dinov2.pth")
    torch.save(model.classifier.state_dict(), "./classifier.pth")

@hydra.main(config_path="./configs", config_name="train_config.yaml") # if hydra version >= 1.3.2 include version_base=None
def main(cfg:DictConfig):
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    if cfg.dataset_name == "FoodSeg103":
        train_data, val_data, classes = load_dataset_foodseg103(dataset_dir=cfg.dataset_dir)
    elif cfg.dataset_name == "Fishency":
        train_data, val_data, classes = load_dataset_fishency(dataset_dir=cfg.dataset_dir)
        print(classes)
        print(f"Total train samples: {len(train_data)}, val samples: {len(val_data)}")
    dist.init_process_group(backend="NCCL")
    rank = dist.get_rank()
    print(f"Start running basic DDP on rank {rank}.")
    train_dataset = CustomDataset(dataset=train_data, transform=train_transform, classes=classes, binary_seg=True) # bak simdi bunu boyle yazdim ya
    test_dataset = CustomDataset(dataset=val_data, transform=val_transform, classes=classes, binary_seg=True)
    # dataloaders
    dist_train_sampler = DistributedSampler(train_dataset, shuffle=True)
    dist_val_sampler = DistributedSampler(test_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_fn, sampler=dist_train_sampler)
    val_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, collate_fn=collate_fn, sampler=dist_val_sampler)
    model = Dinov2ForSemanticSegmentation(cfg, len(classes)).to("cuda")
    # ========================================================================
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"training {cfg.model_name} with {total_trainable_params} parameters")
    split_index = int(len(list(model.named_parameters())) // 1) # freezing half of the dinov2 layers transfer learning
    for name, param in list(model.named_parameters())[:split_index]:
        if name.startswith("dino"):
            param.requires_grad = False
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"training {cfg.model_name} with {total_trainable_params} parameters")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = DataParallel(model, device_ids=[0, 1])
    # ========================================================================
    train_model(model=model, cfg=cfg, train_dataloader=train_dataloader, val_dataloader=val_dataloader, classes=classes)
if __name__ == "__main__":
    main()