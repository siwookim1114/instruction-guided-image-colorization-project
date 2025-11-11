import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
from transformers import AutoTokenizer

class InstructionColorizationDataset(Dataset):
    def __init__(self, img_dir, ann_path, tokenizer, transform = None):
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        with open(ann_path, "r") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)

    def rgb_to_lab(self, img_path):
        """
        Converts an RGB image into normalized L*a*b tensor
        Returns:
            L(torch.Tensor) : shape [1, H, W], range [-1, 1]
            ab (torch.Tensor) : shape [2, H, W], range[-1, 1]
        """
        # Load RGB image and normalize to [0, 1]
        rgb = np.array(Image.open(img_path).convert("RGB"), dtype = np.float32) / 255.0

        # Convert RGB -> L*a*b
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Splitting channels
        L_channel = lab[:, :, 0]   # [0.0, 100.0]
        ab_channels = lab[:, :, 1:] # [-128, + 128]

        # Normalize to [-1, 1]
        L_norm = L_channel / 50.0 - 1.0
        ab_norm = ab_channels / 128.0

        # Converting to torch tensors
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float() # [1, H, W]
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float() # [2, H, W]

        return L_tensor, ab_tensor
    
    def __getitem__(self, idx):
        """Gets filename and corresponding instruction"""
        img_name = list(self.data.keys())[idx]
        instruction_data = self.data[img_name]
        instruction = instruction_data[0] if isinstance(instruction_data, list) else instruction_data
        img_path = os.path.join(self.img_dir, img_name)

        # Convert image to L*a*b tensors
        L, ab = self.rgb_to_lab(img_path)

        # Apply transform (e.g., resize, crop)
        if self.transform:
            L = self.transform(L)
            ab = self.transform(ab)
        
        # Tokenize the instruction
        text_tokens = self.tokenizer(
            instruction,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )
        return {
            "L" : L,        # Grayscale input
            "ab": ab,       # Color target
            "instruction" : instruction,      # Raw text
            "text_input" : text_tokens
        }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = InstructionColorizationDataset(
        img_dir = os.path.abspath("data/raw/train2014"),
        ann_path = os.path.abspath("data/processed/instruction_train2014.json"),
        tokenizer = tokenizer
    )
    sample = dataset[0]
    print("L", sample["L"].shape)
    print("ab", sample["ab"].shape)
    print("instruction", sample["instruction"])

    



        


