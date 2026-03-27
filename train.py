import torch
from loss import contrastive_loss, triplet_loss
from model import ResnetEmbedding
from dataset import ContrastiveDataset, TripletDataset, transform, SimpleDataset, BalancedBatchSampler
import yaml 
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
import os

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
    



def get_hard_triplets(embeddings, labels):
    
    # pairwise eucleadian
    dist_mat = torch.cdist(embeddings, embeddings, p=2)

    labels = labels.unsqueeze(0)  # Shape (1, B)
    mask_pos = (labels == labels.T).bool()  # True where images are same class
    mask_neg = (labels != labels.T).bool()  # True where images are different classes

    # Hard Positives
    # For each anchor, find the same-class image with the MAX distance.
    # ignore the distance to the anchor itself (which is 0).
    dist_mat_pos = dist_mat.clone()
    dist_mat_pos[~mask_pos] = 0  # Ignore different-class distances
    
    # Get indices of maximum distance for each row
    hard_pos_idx = torch.argmax(dist_mat_pos, dim=1)
    hardest_positives = embeddings[hard_pos_idx]

    # Hardest Negatives 
    dist_mat_neg = dist_mat.clone()
    dist_mat_neg[~mask_neg] = float('inf')  # Ignore same-class distances by making them infini
    
    hard_neg_idx = torch.argmin(dist_mat_neg, dim=1)
    hardest_negatives = embeddings[hard_neg_idx]

    return embeddings, hardest_positives, hardest_negatives



def main():
    config = load_config("config.yaml")
    dataset_train = None
    dataset_test = None
    dataset_val = None
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    embedding_dim = config["embedding_dim"]
    chkpnt_path = config["checkpoint_path"]
    lr = config["lr"]
    checkpoint_interval = 5


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
        case "Hard Mining":
            dataset_train = SimpleDataset(train_path, transform=transform)
            dataset_val = SimpleDataset(val_path, transform=transform)
            dataset_test = SimpleDataset(test_path, transform=transform)
        

    #make dataloaders 
    #we use custom smpler for hard mining. 
    if config["dataset"] != "Hard Mining":
        train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    else:
        train_sampler = BalancedBatchSampler(dataset_train, n_classes=8, n_samples=4)
        train = DataLoader(dataset_train, batch_sampler=train_sampler)
        val_sampler = BalancedBatchSampler(dataset_val, n_classes=8, n_samples=4)
        val = DataLoader(dataset_val, batch_sampler=val_sampler)
        test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Load model,loss, optimzer 
    model = ResnetEmbedding(embedding_dim=embedding_dim)
    model = model.to(device)
    
    #since loss is a function, we will just directly use it

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if config["dataset"] == "Contrastive":
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
            
            if (epoch+1) % checkpoint_interval == 0:
                torch.save({
                    "epoch" : epoch, 
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss" : loss.item()
                }, f"{chkpnt_path}/checkpoint_epoch_{epoch}.pth")
    elif config["dataset"] == "Triplet":
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for anch, pos, neg in train:
                optimizer.zero_grad()   #make the gradients 0 for the current batch

                anch = anch.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                #collect the embedding for the images
                anch_emb = model(anch)
                pos_emb = model(pos)
                neg_emb = model(neg)

               
                loss = triplet_loss(anch_emb, pos_emb, neg_emb)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for anch, pos, neg in val:
                    anch = anch.to(device)
                    pos = pos.to(device)
                    neg = neg.to(device)
                    

                    anch_emb = model(anch)
                    pos_emb = model(pos)
                    neg_emb = model(neg)
                    loss = triplet_loss(anch_emb, pos_emb, neg_emb)

                    val_loss += loss.item()
            val_loss /= len(val)



            print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}")
            
            if (epoch+1) % checkpoint_interval == 0:
                torch.save({
                    "epoch" : epoch, 
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss" : loss.item()
                }, f"{chkpnt_path}/checkpoint_epoch_{epoch}.pth")
    elif config["dataset"] == "Hard Mining":
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for img, label in train:
                optimizer.zero_grad()   #make the gradients 0 for the current batch

                img = img.to(device)
                label_tensor = torch.tensor(label).to(device)
                #collect the embedding for the images of the entire batch
                img_emb = model(img)
                anch_emb, pos_emb, neg_emb = get_hard_triplets(img_emb, label_tensor) 
                loss = triplet_loss(anch_emb, pos_emb, neg_emb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img, label in val:
                    img = img.to(device)
                    label_tensor = torch.tensor(label).to(device)
                
                    img_emb = model(img)
                    anch_emb, pos_emb, neg_emb = get_hard_triplets(img_emb, label_tensor) 
                    loss = triplet_loss(anch_emb, pos_emb, neg_emb)

                    val_loss += loss.item()
            val_loss /= len(val)



            print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}")
            
            if (epoch+1) % checkpoint_interval == 0:
                torch.save({
                    "epoch" : epoch, 
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss" : loss.item()
                }, f"{chkpnt_path}/checkpoint_epoch_{epoch}.pth")







if __name__ == "__main__":
    main()

