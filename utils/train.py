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
            loss = criterion(pred_ab, ab)

            # Backward Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

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
       model, dataloader, optimizer, device, num_epochs = 10, save_dir = "checkpoints"
   )
