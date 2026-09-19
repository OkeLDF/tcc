import os
import time
from numbers import Number
import torch
from pathlib import Path

from log import logger  # local log.py with global logger


def print_message(msg, end='\n'):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + str(msg), end=end)


def time_diff(start_time, end_time):
    elapsed = end_time - start_time
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    
    return f'{hours:.0f}h {minutes:.0f}m {seconds:.0f}s'
    

def epoch_stats(epoch: int, history_entry: dict):
    log_string = f'Epoch: {epoch:>3}  |  '

    metrics_string = ', '.join([
        f'{name}: {metric}'
        for name, metric in history_entry.items()
    ])

    logger.info(log_string + metrics_string)


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj
    

def _mem(tag):
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    logger.info(f"[{tag}] allocated={a:.2f} GB reserved={r:.2f} GB")
    print(f"[{tag}] allocated={a:.2f} GB reserved={r:.2f} GB")
    
    
def load_checkpoint(model, path, device, optimizer=None, scheduler=None):
    """Restore a checkpoint written by utils.save_checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state'])  # strict=True on purpose

    if optimizer is not None and checkpoint.get('optimizer_state') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    if scheduler is not None and checkpoint.get('scheduler_state') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    logger.info(f'Loaded checkpoint from {path} (epoch {epoch}, metrics {metrics})')
    return checkpoint


def save_checkpoint(model, optimizer, scheduler, epoch: int, phase: str, metrics: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "epoch": epoch,
        "phase": phase,
        "model_state": _to_cpu(model.state_dict()),
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "scheduler_state": _to_cpu(scheduler.state_dict()) if scheduler is not None else None,
        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, Number)},
    }, tmp)
    os.replace(tmp, path)
    torch.cuda.empty_cache()
    print(f"  checkpoint → {path}")


class EarlyStopping:
    """
    Stops training when monitored metric stops improving.

    Args:
        patience:   epochs to wait after last improvement before stopping
        min_delta:  minimum change to qualify as an improvement
        mode:       'min' for loss, 'max' for metrics like AUROC/MCC
    """
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def step(self, score: float) -> bool:
        """
        Call at the end of each epoch.
        Returns True if training should stop.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
