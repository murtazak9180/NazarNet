import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from model import ResnetEmbedding
from dataset import SimpleDataset, transform

base_dir = Path.home() / "Documents" / "University" / "DL" / "Murtaza_bscs22030_03"
weights_dir = base_dir / "weights"
output_base = base_dir / "embeddings"
csv_dir = base_dir / "dataset_split"

models_to_process = [
    {"name": "contrastive", "weight_file": "contrastive.pth"},
    {"name": "triplet", "weight_file": "triplet.pth"}, # Corrected extension from  .pth
    {"name": "hardmining", "weight_file": "hardmine.pth"}
]

splits = ["train", "val", "test"]

def save_embeddings_for_model(model, dataloader, save_path, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            # Generate normalized embedding vectors
            embeddings = model(imgs)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Concatenate and save as NumPy files 
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)
    
    np.save(save_path / "embeddings.npy", all_embeddings)
    np.save(save_path / "labels.npy", all_labels)
    print(f"Saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_dim = 128 # As per assignment spec 

    for model_info in models_to_process:
        print(f"Processing model: {model_info['name']}")
        
        # Load Model and Weights
        model = ResnetEmbedding(embedding_dim=embedding_dim)
        weight_path = weights_dir / model_info['weight_file']
        
        if not weight_path.exists():
            print(f"Warning: {weight_path} not found. Skipping...")
            continue
            
        checkpoint = torch.load(weight_path, map_location=device)
        # Handle both full checkpoints and state_dicts
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(device)

        for split in splits:
            # Setup Paths 
            csv_path = csv_dir / f"{split}.csv"
            save_path = output_base / model_info['name'] / split
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Use SimpleDataset for all splits to get 1:1 image-to-embedding mapping 
            dataset = SimpleDataset(csv_path, transform=transform)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            save_embeddings_for_model(model, loader, save_path, device)

if __name__ == "__main__":
    main()