from dataset.dataset import CustomDataset, load_dataset_foodseg103, create_cmap, check_dataset, collate_fn
from dataset.visualize_dataset import visualize_map
import torch
from torchvision.transforms import transforms
import hydra
from omegaconf import DictConfig, OmegaConf
from models.dino_seg import Dinov2ForSemanticSegmentation
from tqdm.auto import tqdm
from models.utils import compute_iou, compute_mean_accuracy
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize(size=(448, 448), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
])
val_transform = transforms.Compose([
    transforms.Resize(size=(448, 448), antialias=True),
])

def train_model(model, cfg, train_dataloader, classes):
    epochs = cfg.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    model.to(device)

    model.train()

    for epoch in range(epochs):
        print("Epoch:", epoch)
        for idx, batch in enumerate(tqdm(train_dataloader)):
            imgs = batch["images"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(imgs, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            with torch.no_grad():
                predicted = outputs.logits.argmax(dim=1)
                predicted_process = predicted[0].detach().cpu().numpy().squeeze()
                plt.imsave("./processing_predicted.png", predicted_process)
                # print(f"type of predictions: {predicted.detach().cpu().numpy().shape}")
                # print(f"type of labels{labels.detach().cpu().numpy().shape}")
        miou = compute_iou(predicted, labels)
        macc = compute_mean_accuracy(predicted, labels, len(classes))
        print("Loss:", loss.item())
        print("Mean_iou:", miou)
        print("Mean accuracy:", macc)
        print("--------------------------------------------------")
    torch.save(model.state_dict(), "./model.pth")

@hydra.main(config_path="./configs", config_name="train_config.yaml", version_base=None)
def main(cfg:DictConfig):
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    train_data, val_data, classes = load_dataset_foodseg103(dataset_dir=cfg.dataset_dir)
    train_dataset = CustomDataset(dataset=train_data, transform=train_transform, classes=classes) # bak simdi bunu boyle yazdim ya
    test_dataset = CustomDataset(dataset=val_data, transform=val_transform, classes=classes)
    colors = create_cmap(classes=classes)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)
    model = Dinov2ForSemanticSegmentation(cfg, len(classes))
    # print(f"categories: {classes}")
    for name, param in model.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"training {cfg.model_name} with {total_trainable_params} parameters")
    train_model(model=model, cfg=cfg, train_dataloader=train_dataloader, classes=classes)
if __name__ == "__main__":
    main()