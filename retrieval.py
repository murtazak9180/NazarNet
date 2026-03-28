import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from PIL import Image

base_dir = Path.home() / "Documents" / "University" / "DL" / "Murtaza_bscs22030_03"
csv_path_test = base_dir / "dataset_split" / "test.csv"

def load_data(model_name, split="test"):
    base_path = base_dir / "embeddings" / model_name / split
    embeddings = np.load(base_path / "embeddings.npy")
    labels = np.load(base_path / "labels.npy")
    return embeddings, labels

def recall_at_k(embeddings, labels, k=1):
    neigh = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    neigh.fit(embeddings)
    _, indices = neigh.kneighbors(embeddings)
    
    correct = 0
    for i in range(len(embeddings)):
        neighbor_labels = labels[indices[i, 1:]]
        if labels[i] in neighbor_labels:
            correct += 1
    return correct / len(embeddings)

def show_retrieval(model_name, query_indices, test_df, k=5):
    embeddings, labels = load_data(model_name, "test")
    neigh = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    neigh.fit(embeddings)
    
    fig, axes = plt.subplots(len(query_indices), k + 1, figsize=(20, 4 * len(query_indices)))
    
    for row, q_idx in enumerate(query_indices):
        dist, indices = neigh.kneighbors(embeddings[q_idx].reshape(1, -1))
        
        query_img = Image.open(test_df.iloc[q_idx]['Path'])
        axes[row, 0].imshow(query_img)
        axes[row, 0].set_title(f"Query: {test_df.iloc[q_idx]['Label']}")
        axes[row, 0].axis('off')
        
        for i in range(1, k + 1):
            n_idx = indices[0][i]
            n_img = Image.open(test_df.iloc[n_idx]['Path'])
            is_correct = test_df.iloc[n_idx]['Label'] == test_df.iloc[q_idx]['Label']
            
            color = 'green' if is_correct else 'red'
            
            axes[row, i].imshow(n_img)
            axes[row, i].set_title(f"Top-{i}: {test_df.iloc[n_idx]['Label']}", color=color, fontsize=10)
            axes[row, i].axis('off')
            
    plt.tight_layout()
    output_path = base_dir / "graphs" / f"{model_name}_visual_retrieval.png"
    plt.savefig(output_path)
    print(f"Retrieval grid saved to {output_path}")

if __name__ == "__main__":
    test_df = pd.read_csv(csv_path_test)
    
    models = ["contrastive", "triplet", "hardmining"]
    print(f"{'Model':<15} | {'Recall@1':<10} | {'Recall@5':<10}")
    print("-" * 40)
    
    for m in models:
        try:
            emb, lbl = load_data(m, "test")
            r1 = recall_at_k(emb, lbl, k=1)
            r5 = recall_at_k(emb, lbl, k=5)
            print(f"{m:<15} | {r1:<10.4f} | {r5:<10.4f}")
        except FileNotFoundError:
            print(f"{m:<15} | Embeddings not found. Run save_embeddings.py first.")

    sample_indices = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800]
    
    print("\nGenerating 10-query retrieval grid for Triplet model...")
    show_retrieval("hardmining", sample_indices, test_df)