import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from transformers import ViTModel

class LoRAViT(nn.Module):

    def __init__(self, r=8, alpha=16, dropout=0.0, device='cuda'):
        super().__init__()
        
        base_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            add_pooling_layer=False
        ).to(device)
        
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["query", "value"],
            lora_dropout=dropout,
        )
        
        self.encoder = get_peft_model(base_encoder, config)
        print('LoRA trainable parameters:')
        self.encoder.print_trainable_parameters()

        projector = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 128)).to(device)

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projector(cls_token)