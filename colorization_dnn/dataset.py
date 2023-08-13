import numpy as np

dataset_dir_grayscale  = "/home/emir/Desktop/dev/myResearch/dataset/l/gray_scale.npy"
dataset_dir_lab = ["/home/emir/Desktop/dev/myResearch/dataset/ab/ab/ab1.npy", "/home/emir/Desktop/dev/myResearch/dataset/ab/ab/ab2.npy", "/home/emir/Desktop/dev/myResearch/dataset/ab/ab/ab3.npy"]



def concat_dataset():
    combined_dataset = []
    for dir in dataset_dir_lab:
        print(dir)
        combined_dataset.append(np.load(dir))
    combined_dataset = np.concatenate(combined_dataset, axis=0)
    print(len(combined_dataset))
    np.save('/home/emir/Desktop/dev/myResearch/dataset/colorization_lab.npy', combined_dataset)
concat_dataset()
        