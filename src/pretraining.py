import torch
import torch.nn as nn
import torch.nn.functional as F

from log import logger
from utils import print_message


LOG_LOSS_EVERY = 512
MAX_CLIP_NORM = 1.0


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z):
        N = z.shape[0] // 2
        z = F.normalize(z, dim=1)

        similarities = (z @ z.T) / self.temperature

        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        targets = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(0, N)
        ]).to(z.device)

        similarities.masked_fill_(mask, float('-inf'))

        loss = F.cross_entropy(similarities, targets)
        return loss



def train_step(encoder, projector, loader, contrastive_loss, optimizer, scheduler=None, device:str='cuda', accumulation_steps=1, autocast_dtype=None, scaler=None):
    device_type = torch.device(device).type
    total_loss = torch.tensor(0.0, device=device)
    len_loader = len(loader)

    if len_loader % accumulation_steps != 0:
        logger.warning(f'loader length is not divisible by accumulation_steps: mod={len_loader % accumulation_steps}')

    encoder.train()
    projector.train()
    optimizer.zero_grad(set_to_none=True)

    for it, batch in enumerate(loader, 1):
        aug_i = batch['augmented_i'].to(device)
        aug_j = batch['augmented_j'].to(device)
        pixel_values = torch.cat([aug_i, aug_j], dim=0)

        if autocast_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                cls_embedding = encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
                projection = projector(cls_embedding)
                loss = contrastive_loss(projection)
            scaler.scale(loss / accumulation_steps).backwards()
        else:
            cls_embedding = encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            projection = projector(cls_embedding)
            loss = contrastive_loss(projection)
            (loss / accumulation_steps).backward()

        total_loss += loss.detach()

        if it % accumulation_steps == 0 or it == len_loader:
            if autocast_dtype is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(..., error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(..., error_if_nonfinite=True)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        print_message(f'[train]: it {it}/{len_loader}', end='\r')
        if it % LOG_LOSS_EVERY == 0:
            print_message(f'[train]: loss at it {it}: {total_loss.item()/it:.4f}', end='\n')


    return total_loss.item() / len_loader


@torch.inference_mode()
def eval_step(encoder, projector, loader, contrastive_loss, device: str = 'cuda', autocast_dtype=None):
    device_type = torch.device(device).type
    total_loss = torch.tensor(0.0, device=device)
    len_loader = len(loader)

    encoder.eval()
    projector.eval()

    for it, batch in enumerate(loader, 1):
        aug_i = batch['augmented_i'].to(device)
        aug_j = batch['augmented_j'].to(device)
        pixel_values = torch.cat([aug_i, aug_j], dim=0)

        if autocast_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                cls_embedding = encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
                projection = projector(cls_embedding)
                loss = contrastive_loss(projection)
        else:
            cls_embedding = encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            projection = projector(cls_embedding)
            loss = contrastive_loss(projection)

        total_loss += loss

        print_message(f'[ eval]: it {it}/{len_loader}', end='\r')
        if it % LOG_LOSS_EVERY == 0:
            print_message(f'[ eval]: loss at it {it}: {total_loss.item()/it:.4f}', end='\n')

    return total_loss.item() / len_loader
