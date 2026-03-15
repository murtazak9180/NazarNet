import os 
from pathlib import Path
import random
import csv


home = Path.home()
dataset_path = home / "Documents" / "University" / "DL" / "dataset_3" / "caltech-101"
dataset_path  = Path(dataset_path)


objects = os.listdir(dataset_path)
objects.remove("BACKGROUND_Google")  #we dont need this folder. its of no use. 

#print(objects)


def construct_csvs(train_imgs, test_imgs, val_imgs, label):
    train_data = [{"Path": dataset_path / label / p, "Label": label} for p in train_imgs]
    val_data = [{"Path": dataset_path / label / p, "Label": label} for p in val_imgs]
    test_data = [{"Path": dataset_path / label / p, "Label": label} for p in test_imgs]

    fields = ["Path", "Label"]
    filenames = ['./dataset_split/train.csv', './dataset_split/test.csv', './dataset_split/val.csv']
    for file in filenames:
        with open(file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if file == './dataset_split/train.csv':
                writer.writerows(train_data)
            elif file == './dataset_split/test.csv':
                writer.writerows(test_data)
            elif file == './dataset_split/val.csv':
                writer.writerows(val_data)
          
            

for obj in objects:
    object_path = dataset_path / obj
    images = os.listdir(object_path)
    num_images = len(images)  #total number of images in the given folder
    num_train_imgs = round(0.70*num_images)  #70% of those for train
    num_val_imgs = round(0.15*num_images) # 15% for val
    num_test_imgs = num_images - num_train_imgs - num_val_imgs #rest for test

    num_indexes = range(num_images)
    train_idxs = random.sample(num_indexes, k=num_train_imgs)  #randomly sample indexes for train
    train_images = [images[i] for i in train_idxs]
    idx_set = set(train_idxs)
    remaining = [val for i, val in enumerate(images) if i not in idx_set]  #extract the remaining images

    num_indexes = range(len(remaining))
    val_idxs = random.sample(num_indexes, k=num_val_imgs)
    val_images = [remaining[i] for i in val_idxs]
    idx_set = set(val_idxs)
    test_images = [val for i, val in enumerate(remaining) if i not in idx_set]  #anything that remains is our test set
    
    construct_csvs(train_images, test_images, val_images, obj)










    



