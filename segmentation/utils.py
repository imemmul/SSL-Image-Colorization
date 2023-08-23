import os
import shutil
def __split_dataset(data_dir, dest, split_ratio):
    split_point = int(len(os.listdir(data_dir)) * split_ratio)
    _ = os.listdir(data_dir)[:split_point]
    val_dirs = os.listdir(data_dir)[split_point:]
    for dir in val_dirs:
        src_img_dir = os.path.join(data_dir, dir)
        dest_img_dir = os.path.join(dest, dir)
        shutil.move(src_img_dir, dest_img_dir)
        print(f"{src_img_dir} moved to {dest_img_dir}")
        


if __name__ == "__main__":
    data_dir_train = "/home/emir/Desktop/dev/myResearch/dataset/VOCStego/imgs/train/"
    data_dir_val = "/home/emir/Desktop/dev/myResearch/dataset/VOCStego/imgs/val/"
    label_dir_train = "/home/emir/Desktop/dev/myResearch/dataset/VOCStego/labels/train/"
    label_dir_val = "/home/emir/Desktop/dev/myResearch/dataset/VOCStego/labels/val/"
    __split_dataset(data_dir=data_dir_train, dest=data_dir_val, split_ratio=0.85)
    __split_dataset(data_dir=label_dir_train, dest=label_dir_val, split_ratio=0.85)