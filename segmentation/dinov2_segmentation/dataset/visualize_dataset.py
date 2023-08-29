import matplotlib.pyplot as plt
import numpy as np



def visualize_map(image, segmentation_map, colors):
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in colors.items():
        color_seg[segmentation_map == label, :] = color

    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)