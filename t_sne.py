import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import os

base_dir = Path.home() / "Documents" / "University" / "DL" / "Murtaza_bscs22030_03"
models = ["contrastive", "triplet", "hardmining"]
save_dir = base_dir / "graphs"

def generate_tsne_plots():
    if not save_dir.exists():
        os.makedirs(save_dir)

    for model_name in models:
        print(f"Generating t-SNE for {model_name}...")
        
        try:
            emb_path = base_dir / "embeddings" / model_name / "test" / "embeddings.npy"
            lbl_path = base_dir / "embeddings" / model_name / "test" / "labels.npy"
            
            embeddings = np.load(emb_path)
            labels = np.load(lbl_path)
        except FileNotFoundError:
            print(f"Skipping {model_name}: Embeddings not found.")
            continue

        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=labels, cmap='tab20', s=10, alpha=0.7)
        
        plt.colorbar(scatter, label='Class ID')
        plt.title(f"t-SNE Visualization: {model_name.capitalize()} Model", fontsize=14)
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.grid(True, linestyle='--', alpha=0.3)

        output_file = save_dir / f"{model_name}_tsne.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    generate_tsne_plots()