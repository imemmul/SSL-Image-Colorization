"""
    Converts given rgb dataset to grayscale
    data_dir = "/home/emir/Desktop/dev/myResearch/dataset/oxford_pet/rgb_images"
    out_dir = "/home/emir/Desktop/dev/myResearch/dataset/oxford_pet/gray_images"
"""
import argparse
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="../")
    parser.add_argument('--out-dir', type=str, default="../")
    args = parser.parse_args()
    return args

def convert_data_gray(data_dir:str, out_dir:str):
    for img_dir in os.listdir(data_dir):
        try:
            if img_dir.endswith("jpg"):
                image = cv2.imread(os.path.join(data_dir, img_dir), 0)
                print(f"img_dir: {img_dir}: {image.shape}")
                cv2.imwrite(os.path.join(out_dir, img_dir), image)
        except Exception as e:
            print(f"{e} while processing image: {os.path.join(data_dir, img_dir)}")

def check_dataset(data_dir):
    for img_dir in os.listdir(data_dir):
        try:
            image = cv2.imread(os.path.join(data_dir, img_dir), 0)
            print(f"img_dir: {img_dir}: {image.shape}")
        except Exception as e:
            print(f"{e} while processing image: {os.path.join(data_dir, img_dir)}")

def remove_nones(rgb_data, gray_data):
    for img_dir in os.listdir(rgb_data):
        if not os.path.exists(os.path.join(gray_data, img_dir)):
            print(f"{img_dir} doesn't exists.")
            os.remove(os.path.join(rgb_data, img_dir))
    print(len(os.listdir(args.data_dir)))
    print(len(os.listdir(args.out_dir)))

if __name__ == "__main__":
    args = parse_args()
    print(len(os.listdir(args.data_dir)))
    print(len(os.listdir(args.out_dir)))
    # convert_data_gray(args.data_dir, args.out_dir)
    # check_dataset(args.data_dir)
    remove_nones(rgb_data=args.data_dir, gray_data=args.out_dir)