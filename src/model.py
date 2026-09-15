import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from transformers import ViTModel, ViTConfig, ViTForImageClassification

class LoRAViTModel(nn.Module):

    def __init__(self, r=8, alpha=16, dropout=0.0, device='cuda'):
        super().__init__()
        
        base_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            add_pooling_layer=False
        ).to(device)
        
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules="all-linear",
            lora_dropout=dropout,
        )
        
        self.encoder = get_peft_model(base_encoder, config)
        print('LoRA trainable parameters:')
        self.encoder.print_trainable_parameters()

        self.projector = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 128)).to(device)

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projector(cls_token)


class LoRAViTClassifier(nn.Module):

    def __init__(self, lora_vit_model, classes, device='cuda'):
        super().__init__()

        self.encoder = lora_vit_model
        self.classes = classes

        self.classifier = nn.Linear(128, len(classes))
        self.classifier.to(device)

        self.config = ViTConfig.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=len(classes),
            id2label=dict(enumerate(classes)),
            label2id={label: i for i, label in enumerate(classes)},
        )

    def forward(self, pixel_values):
        embeddings = self.encoder(pixel_values).logits
        logits = self.classifier(embeddings)
        return logits