from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import random

class ContrastiveDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.df = pd.read_csv(csv_path)
        self.transform = transform

        self.paths = self.df['Path'].values
        self.labels = self.df['Label'].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img1_path = self.paths[idx]
        img1_label = self.labels[idx]

        if random.random() > 0.5:
            candidates = self.df[self.df['Label'] == img1_label]   #sample randomly from the same class. 
            img2_path = candidates.sample(n=1)['Path'].values[0]
            label = 1
        else:
            candidates = self.df[self.df['Label'] != img1_label]   #sample randomly from a class with different label. 
            img2_path = candidates.sample(n=1)['Path'].values[0]
            label = 0

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label



class TripletDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.paths = self.df['Path'].values
        self.labels = self.df['Label'].values

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        anchor_img_pth = self.paths[idx]
        anchor_img_lbl = self.labels[idx]

        positive_img_candidates = self.df[(self.df['Label'] == anchor_img_lbl) & (self.df['Path'] != anchor_img_pth)]  #belong to the same class and are not the anchor image
        positive_image_pth = positive_img_candidates.sample(n=1)['Path'].values[0]

        negative_img_candidates = self.df[self.df['Label'] != anchor_img_lbl]
        negative_image_pth = negative_img_candidates.sample(n=1)['Path'].values[0]

        anch_img = Image.open(anchor_img_pth)
        pos_img = Image.open(positive_image_pth)
        neg_img = Image.open(negative_image_pth)

        if self.transform:
            anch_img = self.transform(anch_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anch_img, pos_img, neg_img



