import torch
from loss import contrastive_loss
from model import ResnetEmbedding
from dataset import ContrastiveDataset, TripletDataset, transform
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
            dataset_train = ContrastiveDataset(csv_path=train_path, transform=transform)
            dataset_val = ContrastiveDataset(csv_path=val_path, transform=transform)
            dataset_test = ContrastiveDataset(csv_path=test_path, transform=transform)
        case "Triplet":
            dataset_train = TripletDataset(train_path, transform=transform)
            dataset_val = TripletDataset(csv_path=val_path, transform=transform)
            dataset_test = TripletDataset(csv_path=test_path, transform=transform)
        

    #make dataloaders 
    train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Load model,loss, optimzer 
    model = ResnetEmbedding(embedding_dim=embedding_dim)
    model = model.to(device)
    
    #since loss is a function, we will just directly use it

    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for img1, img2, label in train:
            optimizer.zero_grad()   #make the gradients 0 for the current batch

            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)
            label = label.float()   #loss expects label to be float, not int

            #collect the embedding for the images
            emb1 = model(img1)
            emb2 = model(img2)

            loss = None 
            if config["dataset"] == "Contrastive":
                loss = contrastive_loss(emb1, emb2, label)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val:
                img1 = img1.to(device)
                img2 = img2.to(device)
                label = label.to(device)
                label = label.float()

                emb1 = model(img1)
                emb2 = model(img2)

                loss = None 
                if config["dataset"] == "Contrastive":
                    loss = contrastive_loss(emb1, emb2, label)

                val_loss += loss.item()
        val_loss /= len(val)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Val Loss: {val_loss:.4f}")






if __name__ == "__main__":
    main()

