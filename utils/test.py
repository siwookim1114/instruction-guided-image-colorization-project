import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import AutoTokenizer
import json
from PIL import Image
import cv2

# Load utils and models
from utils.preprocess import InstructionColorizationDataset
from models.text_encoder import TextEncoder
from models.unet_encoder import UNetEncoder
from models.colorization_model import InstructionColorizationModel
from torchvision import transforms
import torch.nn.functional as F

def lab_to_rgb(L, ab):
    """
    Convert LAB tensor to RGB numpy array
    L: (1, H, W) or (H, W)
    ab: (2, H, W) or (H, W, 2)
    """
    if isinstance(L, torch.Tensor):
        L = L.detach().cpu().numpy()
    if isinstance(ab, torch.Tensor):
        ab = ab.detach().cpu().numpy()
    
    # Reshape if needed
    if len(L.shape) == 3:
        L = L.squeeze(0)  # (H, W)
    if len(ab.shape) == 3 and ab.shape[0] == 2:
        ab = np.transpose(ab, (1, 2, 0))  # (H, W, 2)
    
    # Create LAB image
    
    # Undo normalization
    L = (L + 1.0) * 50.0
    ab = ab * 128.0

    lab = np.concatenate([L[..., np.newaxis], ab], axis=-1)
    
    # Convert to RGB
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    return rgb

def visualize_results(L, pred_ab, gt_ab, text_instruction, save_path=None):
    """
    Visualize results: Grayscale input, predicted colorization, ground truth
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Grayscale input
    axes[0].imshow(L.squeeze(), cmap='gray')
    axes[0].set_title('Grayscale Input')
    axes[0].axis('off')
    
    # Predicted colorization
    pred_rgb = lab_to_rgb(L, pred_ab)
    axes[1].imshow(pred_rgb)
    axes[1].set_title('Predicted Colorization')
    axes[1].axis('off')
    
    # Ground truth
    gt_rgb = lab_to_rgb(L, gt_ab)
    axes[2].imshow(gt_rgb)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    # Add instruction text
    plt.suptitle(f"Instruction: {text_instruction}", fontsize=12, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def evaluate_model(model, dataloader, device, save_dir="test_results"):
    """
    Evaluate model on test dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "visualizations"), exist_ok=True)
    
    model.eval()
    criterion = nn.SmoothL1Loss()
    
    # Metrics storage
    losses = []
    mse_scores = []
    mae_scores = []
    psnr_scores = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
            L = batch["L"].to(device)
            ab = batch["ab"].to(device)
            text_input = {k: v.to(device) for k, v in batch["text_input"].items()}
            text_instruction = batch.get("instruction", [""] * len(L))
            
            # Forward pass
            pred_ab = model(L, text_input)
            
            # Calculate loss
            loss = criterion(pred_ab, ab)
            losses.append(loss.item())
            
            # Calculate MSE and MAE
            pred_np = pred_ab.cpu().numpy().flatten()
            gt_np = ab.cpu().numpy().flatten()
            
            mse = mean_squared_error(gt_np, pred_np)
            mae = mean_absolute_error(gt_np, pred_np)
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            
            # Calculate PSNR
            mse_per_channel = F.mse_loss(pred_ab, ab, reduction='none').mean(dim=[1,2,3])
            psnr = 10 * torch.log10(1.0 / mse_per_channel)
            psnr_scores.extend(psnr.cpu().numpy())
            
            # Visualize first few samples
            if idx < 5:  # Visualize first 5 batches
                for i in range(min(2, len(L))):  # Visualize first 2 samples per batch
                    save_path = os.path.join(
                        save_dir, 
                        "visualizations", 
                        f"sample_{idx}_{i}.png"
                    )
                    visualize_results(
                        L[i].cpu().numpy(),
                        pred_ab[i].cpu().numpy(),
                        ab[i].cpu().numpy(),
                        text_instruction[i] if i < len(text_instruction) else "",
                        save_path
                    )
    
    # Calculate average metrics
    avg_loss = np.mean(losses)
    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_psnr = np.mean(psnr_scores)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average Loss: {avg_loss:.5f}")
    print(f"Average MSE: {avg_mse:.5f}")
    print(f"Average MAE: {avg_mae:.5f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Number of samples: {len(dataloader.dataset)}")
    print("="*50)
    
    # Save metrics to file
    metrics = {
        "loss": float(avg_loss),
        "mse": float(avg_mse),
        "mae": float(avg_mae),
        "psnr": float(avg_psnr),
        "num_samples": len(dataloader.dataset)
    }
    
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def load_checkpoint(model, checkpoint_path, device):
    """
    Load model checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return model

def test_single_image(model, image_path, instruction, device, tokenizer, img_size=(480, 640)):
    """
    Test model on a single image with instruction
    """
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    # Read image
    img = Image.open(image_path).convert('RGB')
    
    # Convert RGB to LAB
    img_np = np.array(img)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    # Normalize
    L = img_lab[:, :, 0] / 100.0  # L channel [0, 100] -> [0, 1]
    ab = img_lab[:, :, 1:] / 128.0  # ab channels [-128, 127] -> [-1, 1]
    
    # Resize if needed
    L_tensor = torch.FloatTensor(L).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    if img_size:
        L_tensor = F.interpolate(L_tensor, size=img_size, mode='bilinear', align_corners=False)
    
    # Tokenize instruction
    text_input = tokenizer(
        instruction,
        return_tensors="pt",
        padding="max_length",
        max_length=77,  # Adjust based on your model
        truncation=True
    )
    
    # Move to device
    L_tensor = L_tensor.to(device)
    text_input = {k: v.to(device) for k, v in text_input.items()}
    
    # Generate prediction
    with torch.no_grad():
        pred_ab = model(L_tensor, text_input)
    
    # Convert to numpy
    L_np = L_tensor[0, 0].cpu().numpy()
    pred_ab_np = pred_ab[0].cpu().numpy()
    
    # Visualize
    pred_rgb = lab_to_rgb(L_np, pred_ab_np)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(L_np, cmap='gray')
    axes[0].set_title('Grayscale Input')
    axes[0].axis('off')
    
    axes[1].imshow(pred_rgb)
    axes[1].set_title(f'Colorized\nInstruction: {instruction}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred_rgb

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load test dataset
    test_dataset = InstructionColorizationDataset(
        img_dir="data/raw/val/val2014",  # Assuming validation/test images
        ann_path="data/processed/instructions_val2014.json",  # Test annotations
        tokenizer=tokenizer,
        transform=transforms.Compose([
            transforms.Resize((480, 640))
        ])
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,  # Don't shuffle for consistent evaluation
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model (same architecture as training)
    model = InstructionColorizationModel(
        text_model="bert-base-uncased",
        text_dim=512,
        base_channels=64
    ).to(device)
    
    # Load trained checkpoint (use your best checkpoint)
    checkpoint_path = "checkpoints/epoch_2.pth"  # Adjust to your best model
    model = load_checkpoint(model, checkpoint_path, device)
    
    # Evaluate on test set
    print("Starting evaluation on test set...")
    metrics = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        save_dir="test_results"
    )

    print("\nTesting completed!")