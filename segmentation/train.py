from models.stego import Stego
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import os
from models.dataset import ContrastiveSegDataset
from models.utils import *
from torchvision import transforms as T

@hydra.main(config_path="configs", config_name="train_config.yml")
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    py_data_dir = cfg.pytorch_data_dir
    data_dir = os.path.join(cfg.output_root, "data")
    log_dir = os.path.join(cfg.output_root, "logs")
    cp_dir = data_dir = os.path.join(cfg.output_root, "cps")
    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cp_dir, exist_ok=True)
    print(data_dir)
    print(cfg.output_root)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()
    
    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=py_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )
    val_loader_crop = "center"
    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=py_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )
    
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    model = Stego(train_dataset.n_classes, cfg)
    tb_logger = TensorBoardLogger(
        os.path.join(log_dir, name),
        default_hp_metric=False
    )
    gpu_args = dict(gpus=-1, accelerator='ddp', val_check_interval=cfg.val_freq)
        # gpu_args = dict(gpus=1, accelerator='ddp', val_check_interval=cfg.val_freq)

    if gpu_args["val_check_interval"] > len(train_loader) // 4:
        gpu_args.pop("val_check_interval")

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(cp_dir, name),
                every_n_train_steps=400,
                save_top_k=2,
                monitor="test/cluster/mIoU",
                mode="max",
            )
        ],
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)
    
if __name__ == "__main__":
    prep_args()
    train()