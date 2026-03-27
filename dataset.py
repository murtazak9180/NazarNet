from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import random
from torchvision import transforms
from torch.utils.data import Sampler
import numpy as np





transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

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
            candidates = self.df[(self.df['Label'] == img1_label) & (self.df['Path'] != img1_path)]   #sample randomly from the same class. 
            img2_path = candidates.sample(n=1)['Path'].values[0]
            label = 1
        else:
            candidates = self.df[self.df['Label'] != img1_label]   #sample randomly from a class with different label. 
            img2_path = candidates.sample(n=1)['Path'].values[0]
            label = 0

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

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

        anch_img = Image.open(anchor_img_pth).convert("RGB")
        pos_img = Image.open(positive_image_pth).convert("RGB")
        neg_img = Image.open(negative_image_pth).convert("RGB")

        if self.transform:
            anch_img = self.transform(anch_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anch_img, pos_img, neg_img



class SimpleDataset(Dataset):  #for hard mining
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        self.class_to_idx = {name: i for i, name in enumerate(sorted(self.df['Label'].unique()))}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['Path']).convert("RGB")
        
        # Convert the string label to its integer ID
        label_id = self.class_to_idx[row['Label']]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label_id 
    


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = np.array(dataset.labels if hasattr(dataset, 'labels') else [dataset[i][1] for i in range(len(dataset))])
        self.classes = np.unique(self.labels)
        self.n_classes = n_classes 
        self.n_samples = n_samples 
        self.batch_size = n_classes * n_samples
        
        # Create a dictionary of indices for each class
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}

    def __iter__(self):
        n_batches = len(self.labels) // self.batch_size
        for _ in range(n_batches):
            # Randomly select 8 classes
            selected_classes = np.random.choice(self.classes, self.n_classes, replace=False)
            batch_indices = []
            for c in selected_classes:
                # Randomly select 4 images from each of those classes
                indices = self.class_indices[c]
                # Use replace=True if a class has fewer than n_samples images
                replace = len(indices) < self.n_samples
                batch_indices.extend(np.random.choice(indices, self.n_samples, replace=replace))
            
            yield batch_indices

    def __len__(self):
        return len(self.labels) // self.batch_size