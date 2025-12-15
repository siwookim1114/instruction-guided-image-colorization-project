import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from transformers import AutoTokenizer

# Load utils and models
from utils.preprocess import InstructionColorizationDataset
from models.text_encoder import TextEncoder
from models.unet_encoder import UNetEncoder
from models.colorization_model import InstructionColorizationModel
from torchvision import transforms

def chroma_loss(pred_ab, gt_ab, eps=1e-6):
    # pred_ab, gt_ab: (B, 2, H, W)
    pred_a, pred_b = pred_ab[:, 0], pred_ab[:, 1]
    gt_a, gt_b     = gt_ab[:, 0], gt_ab[:, 1]

    chroma_pred = torch.sqrt(pred_a ** 2 + pred_b ** 2 + eps)
    chroma_gt   = torch.sqrt(gt_a ** 2 + gt_b ** 2 + eps)

    return torch.mean(torch.abs(chroma_pred - chroma_gt))

def weighted_l2_loss(output_ab, target_ab, weight_power=2.0, epsilon=1e-6):
    # 1. Calculate the standard L2 distance (squared error)
    squared_error = (output_ab - target_ab)**2
    
    # Sum the squared errors for 'a' and 'b' channels for each pixel
    l2_distance = torch.sum(squared_error, dim=1, keepdim=True)
    
    
    # 2. Calculate Saturation (Chroma) for Weighting (S = sqrt(a^2 + b^2))
    target_a = target_ab[:, 0, :, :]
    target_b = target_ab[:, 1, :, :]
    
    # Chroma squared: a^2 + b^2. Shape: (Batch, H, W)
    chroma_sq = target_a**2 + target_b**2
    
    # Chroma (Saturation S): sqrt(a^2 + b^2). Shape: (Batch, H, W)
    chroma = torch.sqrt(chroma_sq + epsilon) # Add epsilon for safety
    
    
    # 3. Define the Weighting Map $\lambda(a, b)$
    weight_map = 1.0 + chroma**weight_power
    
    # The final loss calculation requires the weight_map to have 
    weight_map = weight_map.unsqueeze(1)
    
    
    # 4. Apply the Weighting
    # Weighted error = L2_distance * Weight_Map
    weighted_error = l2_distance * weight_map
    
    # The final loss is the mean of the weighted errors across all pixels and batch
    loss = torch.mean(weighted_error)
    
    return loss

def train(model, dataloader, optimizer, device, num_epochs, save_dir = "checkpoints"):
    os.makedirs(save_dir, exist_ok = True)
    
    criterion = nn.SmoothL1Loss()     # Stabalization for LAB colorization
    model.train()

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        pbar = tqdm(dataloader, desc = "Training", leave = False)

        for batch in pbar:
            L = batch["L"].to(device)          # (B, 1, H, W)
            ab = batch["ab"].to(device)        # (B, 2, H, W)
            text_input = {k: v.to(device) for k, v in batch["text_input"].items()}

            # Forward pass
            pred_ab = model(L, text_input)
            # loss_l1 = criterion(pred_ab, ab)
            # loss_chroma = chroma_loss(pred_ab, ab)
            
            # if epoch < 1:
            #     loss = loss_l1
            # else:
            #     loss = loss_chroma
            loss = weighted_l2_loss(pred_ab, ab)

            # Backward Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({
                # "L1": loss_l1.item(),
                # "chroma": loss_chroma.item(),
                "total": loss.item()
            })

        # Epoch Summary
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.5f}")

        # Saving Checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)

        print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Loading tokenizer for text tokenization
    dataset = InstructionColorizationDataset(
        img_dir = "data/raw/train/train2014",
        ann_path = "data/processed/instructions_train2014.json",
        tokenizer = tokenizer,
        transform=transforms.Compose([transforms.Resize((480,640))])
    )
    dataloader = DataLoader(
        dataset,
        batch_size = 4,
        shuffle = True,
        num_workers = 2,
        pin_memory = True   # CUDA
    )

    # Initialize model
    model = InstructionColorizationModel(
        text_model = "bert-base-uncased",
        text_dim = 512,
        base_channels = 64
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 1e-4,
        weight_decay = 1e-5
    )
    
    # Training
    train(
       model, dataloader, optimizer, device, num_epochs = 8, save_dir = "checkpoints"
   )
