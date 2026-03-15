from torch.utils.data import Dataset
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