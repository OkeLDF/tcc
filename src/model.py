import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from transformers import ViTModel, ViTConfig, ViTForImageClassification

class LoRAViTModel(nn.Module):

    def __init__(self, r=8, alpha=16, dropout=0.0, gradient_checkpointing=False, device='cuda'):
        super().__init__()
        
        base_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            add_pooling_layer=False
        ).to(device)

        # Recomputes each transformer block's activations during backward
        # instead of storing them, so training the LoRA adapters fits in
        # limited GPU memory. Results are unchanged; only compute increases.
        if gradient_checkpointing:
            base_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
        
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules="all-linear",
            lora_dropout=dropout,
        )
        
        self.encoder = get_peft_model(base_encoder, config)

        # PEFT makes the embedding output require grad when checkpointing is
        # on. That is only needed with use_reentrant=True; here it would force
        # a useless backward pass through the frozen encoder in phase 1.
        if gradient_checkpointing:
            base_encoder.disable_input_require_grads()

        print('LoRA trainable parameters:')
        self.encoder.print_trainable_parameters()

        self.projector = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 128)).to(device)

    def forward(self, pixel_values, return_features=False):
        outputs = self.encoder(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :].clone()
        if return_features:
            return cls_token
        return self.projector(cls_token)


class LoRAViTClassifier(nn.Module):

    def __init__(self, lora_vit_model, classes, device='cuda'):
        super().__init__()

        self.encoder = lora_vit_model
        self.classes = classes

        self.classifier = nn.Linear(768, len(classes))
        self.classifier.to(device)

        self.id2label = dict(enumerate(classes))
        self.label2id = {label: i for i, label in enumerate(classes)}

    def forward(self, pixel_values):
        features = self.encoder(pixel_values, return_features=True)
        logits = self.classifier(features)
        return logits
