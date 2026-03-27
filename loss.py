
import torch
margin = 1

#Contrastive loss takes in the 2 embeddings of the given images and the label associated with it and compute 
#loss between them
def contrastive_loss(emb1, emb2, labels):
    #at this point the embeddings are B X D - Batch X Dimensions(number of samples)
    #Labels is of size D
    #We must return average loss across the entire batch
    dist_sq = torch.sum((emb1 - emb2)**2, dim=1)
    dist = torch.sqrt(dist_sq)

    losses = (labels * dist_sq) + ((1 - labels) * torch.clamp(margin - dist, min=0)**2)
    total_loss = torch.mean(losses)
    return total_loss


def triplet_loss(anchor, positive, negative, margin=0.2):
    
    # Calculate Euclidean distances D(a,p) and D(a,n)
    # D = sqrt(sum((x1 - x2)^2))
    dist_ap = torch.sqrt(torch.sum((anchor - positive)**2, dim=1))
    dist_an = torch.sqrt(torch.sum((anchor - negative)**2, dim=1))
    
    losses = torch.clamp(dist_ap - dist_an + margin, min=0.0)
    return torch.mean(losses)