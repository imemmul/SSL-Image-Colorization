# TAKEN FROM https://github.com/facebookresearch/dinov2/blob/add-features-pca-nb/notebooks/patch_features.ipynb

from typing import Tuple
from sklearn.decomposition import PCA
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy.ndimage import binary_closing, binary_opening
import urllib
import io
import os

os.environ["XFORMERS_DISABLED"] = "1"

DEFAULT_SMALLER_EDGE_SIZE = 448
DEFAULT_BACKGROUND_THRESHOLD = 0.05
DEFAULT_APPLY_OPENING = False
DEFAULT_APPLY_CLOSING = False


cp_dir = "/home/doga/Projects/emir_workspace/checkpoints_trained/dinov2_segmenter_0.pth"
MODEL_NAME = "dinov2_vitl14"
REPO_DIR = "facebookresearch/dinov2"

def load_array_from_url(url: str) -> np.ndarray:
    with urllib.request.urlopen(url) as f:
        array_data = f.read()
        g = io.BytesIO(array_data)
        return np.load(g)


def make_foreground_mask(tokens,
                         grid_size: Tuple[int, int],
                         background_threshold: float = 0.0,
                         apply_opening: bool = True,
                         apply_closing: bool = True,
                         standard_array: np.ndarray = None) -> np.ndarray:
    projection = tokens @ standard_array
    mask = projection > background_threshold
    mask = mask.reshape(*grid_size)
    if apply_opening:
        mask = binary_opening(mask)
    if apply_closing:
        mask = binary_closing(mask)
    return mask.flatten()


def render_patch_pca(model,
                     image: Image,
                     smaller_edge_size: float = 448,
                     patch_size: int = 14,
                     background_threshold: float = 0.05,
                     apply_opening: bool = False,
                     apply_closing: bool = False) -> Image:
    image_tensor, grid_size = prepare_image(image, smaller_edge_size, patch_size)

    with torch.inference_mode():
        image_batch = image_tensor.unsqueeze(0)
        tokens = model.get_intermediate_layers(image_batch)[0].squeeze()
    standard_array = load_array_from_url("https://dl.fbaipublicfiles.com/dinov2/arrays/standard.npy")
    mask = make_foreground_mask(tokens,
                                grid_size,
                                background_threshold,
                                apply_opening,
                                apply_closing,
                                standard_array=standard_array)

    pca = PCA(n_components=3)
    pca.fit(tokens[mask])
    projected_tokens = pca.transform(tokens)

    t = torch.tensor(projected_tokens)
    t_min = t.min(dim=0, keepdim=True).values
    t_max = t.max(dim=0, keepdim=True).values
    normalized_t = (t - t_min) / (t_max - t_min)

    array = (normalized_t * 255).byte().numpy()
    array[~mask] = 0
    array = array.reshape(*grid_size, 3)

    return Image.fromarray(array).resize((image.width, image.height), 0)


def make_transform(smaller_edge_size: int) -> transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BICUBIC

    return transforms.Compose([
        transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

def prepare_image(image: Image,
                  smaller_edge_size: float,
                  patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    transform = make_transform(int(smaller_edge_size))
    image_tensor = transform(image)

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size) # h x w (TODO: check)
    return image_tensor, grid_size

def load_cp(dir):
    cp = torch.load(dir)
    return cp['backbone']

def load_model(model_name, repo_dir):
    model = torch.hub.load(repo_or_dir=repo_dir, model=f"{model_name}")
    return model


def main():
    model = load_model(MODEL_NAME, REPO_DIR)
    cp_backbone = load_cp(cp_dir)
    model.load_state_dict(cp_backbone)
    print("model loaded")
    model.eval()
    print(f"patch size: {model.patch_size}")
    example_image = Image.open("/home/doga/Projects/emir_workspace/data/matting-data/FishencyBackgrounds/train/sample49.jpg").convert("RGB")
    standard_array = load_array_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/arrays/standard.npy")
    print(f"standard_array: {standard_array}")
    # pca_img = render_patch_pca(model=model,
    #                             image=example_image,
    #                             smaller_edge_size=DEFAULT_SMALLER_EDGE_SIZE,
    #                             patch_size=model.patch_size,
    #                             background_threshold=DEFAULT_BACKGROUND_THRESHOLD,
    #                             apply_opening=DEFAULT_APPLY_OPENING,
    #                             apply_closing=DEFAULT_APPLY_CLOSING)
    # pca_img.save("./pca_img.png")

if __name__ == "__main__":
    main()