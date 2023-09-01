import matplotlib.pyplot as plt
import numpy as np



def visualize_map(image, segmentation_map, colors):
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    print(colors)
    for label, color in colors.items():
        print(f"label: {label}")
        color_seg[segmentation_map == int(label), :] = color

    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.imsave(fname="./seg_mask.png", arr=img)
    