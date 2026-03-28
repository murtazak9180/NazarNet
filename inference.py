import torch
import argparse
from PIL import Image
from pathlib import Path
from model import ResnetEmbedding
from dataset import transform

def run_inference(image_path, weights_path, embedding_dim=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResnetEmbedding(embedding_dim=embedding_dim)
    
    if not Path(weights_path).exists():
        print(f"Error: Weights file not found at {weights_path}")
        return None

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load and Preprocess Image
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device) 
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Generate Embedding
    with torch.no_grad():
        embedding = model(img_tensor) # Already L2 normalized by the model 
    
    return embedding.cpu().numpy().flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained .pth file.")
    
    args = parser.parse_args()

    vector = run_inference(args.image, args.weights)
    
    if vector is not None:
        print(f"\nSuccessfully generated embedding for: {args.image}")
        print(f"Vector Shape: {vector.shape}")
        print(f"First 5 dimensions: {vector[:5]}")