import torch
from loss import contrastive_loss
from model import ResnetEmbedding
from dataset import ContrastiveDataset, TripletDataset
import yaml 
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim

#Construct the paths for the dataset csvs
home = Path.home()
base_dir = home / "Documents" / "University" / "DL" / "Murtaza_bscs22030_03" / "dataset_split"
train_path = base_dir / "train.csv"
test_path = base_dir / "test.csv"
val_path = base_dir / "val.csv"

def load_config(file_name):
    with open(file_name, "r") as file:
        config = yaml.safe_load(file)
        return config
    


def main():
    config = load_config("config.yaml")
    dataset_train = None
    dataset_test = None
    dataset_val = None
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    embedding_dim = config["embedding_dim"]
    lr = config["lr"]

    #load the relevent dataset
    match config["dataset"]:
        case "Contrastive":
            dataset_train = ContrastiveDataset(csv_path=train_path, transform=)
            dataset_val = ContrastiveDataset(csv_path=val_path, transform=)
            dataset_test = ContrastiveDataset(csv_path=test_path, transform=)
        case "Triplet":
            dataset_train = TripletDataset(train_path, transform=)
            dataset_val = TripletDataset(csv_path=val_path, transform=)
            dataset_test = TripletDataset(csv_path=test_path, transform=)
        case _:

    #make dataloaders 
    train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)


    #Load model,loss, optimzer 
    model = ResnetEmbedding(embedding_dim=embedding_dim)
    
    #since loss is a function, we will just directly use it

    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        for img1, img2, label in train:
            optimizer.zero_grad()   #make the gradients 0 for the current batch

            #collect the embedding for the images
            emb1 = model(img1)
            emb2 = model(img2)

            loss = None 
            if config["dataset"] == "Contrastive":
                loss = contrastive_loss(emb1, emb2, label)
            
            loss.backward()
            optimizer.step()







