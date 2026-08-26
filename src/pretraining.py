import torch
import torch.nn as nn

from log import logger
from utils import print_message


LOG_LOSS_EVERY = 512
MAX_CLIP_NORM = 1.0


def train_step(model, loader, criterion, optimizer, scheduler=None, device:str='cuda', accumulation_steps=1, autocast_dtype=None):
    device_type = torch.device(device).type
    total_loss = torch.tensor(0.0, device=device)
    len_loader = len(loader)

    if len_loader % accumulation_steps != 0:
        logger.warning(f'loader length is not divisible by accumulation_steps: mod={len_loader % accumulation_steps}')

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for it, batch in enumerate(loader, 1):
        pixel_values = batch['pixel_values'].to(device)
        label = batch['label'].to(device)

        if autocast_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                logits = model(pixel_values).logits
                loss = criterion(logits, label)
        else:
            logits = model(pixel_values).logits
            loss = criterion(logits, label)

        total_loss += loss.detach()
        loss = loss / accumulation_steps
        loss.backward()

        if it % accumulation_steps == 0 or it == len_loader:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_CLIP_NORM, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

        print_message(f'[train]: it {it}/{len_loader}', end='\r')
        if it % LOG_LOSS_EVERY == 0:
            print_message(f'[train]: loss at it {it}: {total_loss.item()/it:.4f}', end='\n')


    return total_loss.item() / len_loader


@torch.inference_mode()
def eval_step(model, loader, criterion, metrics=None, device:str='cuda', autocast_dtype=None):
    device_type = torch.device(device).type
    total_loss = torch.tensor(0.0, device=device)
    len_loader = len(loader)
    model.eval()

    for it, batch in enumerate(loader, 1):
        pixel_values = batch['pixel_values'].to(device)
        label = batch['label'].to(device)

        if autocast_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                logits = model(pixel_values).logits
                loss = criterion(logits, label)
        else:
            logits = model(pixel_values).logits
            loss = criterion(logits, label)

        total_loss += loss.detach()

        if metrics is not None:
            probs = logits.softmax(dim=-1)[:, 1].float()
            metrics.update(probs, label)

        print_message(f'[ eval]: it {it}/{len_loader}', end='\r')
        if it % LOG_LOSS_EVERY == 0:
            print_message(f'[ eval]: loss at it {it}: {total_loss.item()/it:.4f}', end='\n')

    return total_loss.item() / len_loader
