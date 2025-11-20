import torch
import torch.nn as nn
from models.text_encoder import TextEncoder
from models.unet_encoder import UNetEncoder

class InstructionColorizationModel(nn.Module):
    def __init__(self, text_model: str, text_dim = 512, base_channels = 64):       # text_dim -> dimension after projection, base_channels -> UNet base channel count
        super().__init__()
        self.text_encoder = TextEncoder(model_name = text_model, output_dim = text_dim)
        self.image_encoder = UNetEncoder(in_channels = 1, base_channels = base_channels)

        # Fusing Text + Visual Features
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 8 + text_dim, base_channels * 8, kernel_size = 1),
            nn.ReLU(inplace = True)
        )

        # Color prediction (Decoding to 2-channel a,b)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(base_channels * 2, 2, 4, 2, 1),  # final 2 channels (a, b)
            nn.Tanh()   # Output in [-1, 1] range as it is normalized
        )

    def forward(self, L, text_inputs):
        """
        Input: 
            L: (B, 1, H, W)
            text_inputs: dict -> "input_ids", "attention_mask"
        """
        # Encoding text -> vector
        text_feat = self.text_encoder(text_inputs)   # shape: (B, text_dim)

        # Encoding image -> UNet features
        x4, _, _, _, _ = self.image_encoder(L) # x4 shape: (B, base*8, H/16, W/16)

        # Injecting text into image feature map
        B, C, H, W = x4.shape
        text_map = text_feat.unsqueeze(-1).unsqueeze(-1)   # (B, text_dim, 1, 1)
        text_map = text_map.expand(-1, -1, H, W)            # (B, text_dim, H, W)

        x_fused = torch.cat([x4, text_map], dim = 1)   # Concatting channels
        x_fused = self.fusion(x_fused)    # (B, base*8, H, W)

        # Decoding -> Predict a,b channels
        ab = self.decoder(x_fused)    # (B, 2, H_full, W_full)
        return ab

        