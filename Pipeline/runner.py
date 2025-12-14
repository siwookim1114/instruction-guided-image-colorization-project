from abc import ABC, abstractmethod

from models.unet_encoder import UNetEncoder
from utils.download_img import Downloader
from utils.generate_instructions import build_instruction_json
from utils.preprocess import InstructionColorizationDataset

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
from utils.test import load_checkpoint, lab_to_rgb
from torchvision import transforms
import torch.nn.functional as F

class Runner(ABC):
    @abstractmethod
    def run(self, inputs: dict = None): ...


class DownloaderRunner(Downloader, Runner):
    def run(self, inputs: dict = None):
        self.download()
        self.extract()


class InstructionRunner(Runner):
    def __init__(self, split_name, annot_path):
        self.split_name = split_name
        self.annot_path = annot_path

    def run(self, inputs: dict = None):
        build_instruction_json(self.annot_path, self.split_name)


class PreprocessorRunner(InstructionColorizationDataset, Runner):
    def run(self, inputs: dict = None):
        return self


class UNetEncoderRunner(UNetEncoder, Runner):
    def run(self, inputs: dict = None):
        self.eval()
        return self

class ExampleRunner(Runner):
    def run(self, inputs: dict = None):
        dataset = inputs["preprocess"]
        encoder = inputs["image_encoder"]

        if not encoder or not dataset:
            raise ValueError("Dependencies have resolved erroneously.")

        print(encoder)
        print(dataset)

        return encoder(dataset[0]["L"].unsqueeze(0))

class GetDeviceRunner(Runner):
    def run(self, inputs: dict=None):
        print("Running GetDeviceRunner")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        return device

class GetTokenizerRunner(Runner):
    def run(self, inputs: dict=None):
        print("Running GetTokenizerRunner")
        return AutoTokenizer.from_pretrained("bert-base-uncased")

class GetInputRunner(Runner):
    def run(self, inputs: dict=None):
        print("Running GetInputRunner")
        print("Input path to image to be colorized: ", end="")
        img_path = input()
        print("Describe the image and color: ", end="")
        prompt = input()
        print("For DEV: which model checkpoint?", end="")
        checkpoint = input()
        try:
            checkpoint = int(checkpoint)
        except ValueError:
            print("taking default checkpoint=1")
            checkpoint = 1
        return (img_path, prompt, checkpoint)
    
class GetModelRunner(Runner):
    def run(self, inputs: dict=None):
        print("Running GetModelRunner")
        device = inputs["GetDevice"]
        _, _, checkpoint_no = inputs["GetInput"]
        checkpoint_path = f"checkpoints_1.0.0/epoch_{checkpoint_no}.pth"
        
        model = InstructionColorizationModel(
            text_model="bert-base-uncased",
            text_dim=512,
            base_channels=64
        ).to(device)
        
        model = load_checkpoint(model, checkpoint_path, device)
        
        return model

class TestSingleImageRunner(Runner):
    def run(self, inputs: dict=None):
        print("Running TestSingleImageRunner")
        device = inputs["GetDevice"]
        image_path, instruction, checkpoint = inputs["GetInput"]
        tokenizer = inputs["GetTokenizer"]
        img_size = (480,640) # fixed for our project
        model = inputs["GetModel"]
        
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
        
        plt.suptitle(f"Instruction: {instruction}", fontsize=12, y=0.95)
        plt.tight_layout()
        plt.savefig("pipeline_sample.png", dpi=150, bbox_inches='tight')
        plt.close()

        return pred_rgb           
    