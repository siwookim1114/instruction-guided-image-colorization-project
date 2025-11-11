import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, List

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 512):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(768, output_dim)   # Reduce dim
    
    def forward(self, text_inputs: Dict[List, List]) -> List:
        """
        text_inputs: dict from tokenizer (input_ids, attention_mask)
        returns: (batch, output_dim)
        """
        outputs = self.encoder(
            input_ids = text_inputs['input_ids'].squeeze(1),
            attention_mask = text_inputs['attention_mask'].squeeze(1)
        )
        cls_embed = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_feat = self.proj(cls_embed)
        return text_feat