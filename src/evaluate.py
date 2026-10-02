"""Detailed test-set evaluation of the fine-tuned classifiers.

For each phase checkpoint (frozen = linear probe, unfrozen = fine-tuning) of
one schema and condition, runs the test set once and writes to PROJECT_LOG:

  eval_{schema}_{cond}_{phase}_predictions.csv  one row per test image
                                               (path, source dataset, label,
                                               prediction, class probabilities)
  eval_{schema}_{cond}_{phase}_per_class.csv    support, precision, recall, F1,
                                               AUROC and AP per class
  eval_{schema}_{cond}_{phase}_per_dataset.csv  support and recall per class
                                               within each source dataset
  eval_{schema}_{cond}_{phase}_confusion.csv    confusion matrix (counts)
  eval_{schema}_{cond}_{phase}_confusion.png    confusion matrix, normalized
                                               by row (true class)

The predictions file holds everything the other files are computed from, so
further analyses don't need the GPU again.
"""

from bisect import bisect_right
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (average_precision_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score)

from model import LoRAViTModel, LoRAViTClassifier
from image_transforms import EvaluationTransform
from step_downstream import _datasets, PROJECT_ROOT
from utils import load_checkpoint
from log import logger


PHASES = ('frozen', 'unfrozen')
DATASET_NAMES = {
    'MendeleyLBCDataset': 'Mendeley LBC',
    'CPSMI2025Dataset': 'CPSMI',
    'BTMDataset': 'BTM',
    'HiCervixDataset': 'HiCervix',
    'PapicitoDataset': 'LUAC',
    'HerlevDataset': 'Herlev',
    'SIPaKMeDDataset': 'SIPaKMeD',
}


def _sample_sources(concat_dataset):
    """Return (path, source dataset name) for each index of the test ConcatDataset."""
    sources = []
    for index in range(len(concat_dataset)):
        part = bisect_right(concat_dataset.cumulative_sizes, index)
        offset = index - (concat_dataset.cumulative_sizes[part - 1] if part else 0)
        remapped = concat_dataset.datasets[part]
        path = remapped.dataset.samples[remapped.indices[offset]][0]
        name = type(remapped.dataset).__name__
        sources.append((path, DATASET_NAMES.get(name, name)))
    return sources


@torch.inference_mode()
def _predict(classifier, loader, device):
    classifier.eval()
    probabilities, labels = [], []
    for it, batch in enumerate(loader, 1):
        logits = classifier(batch['pixel_values'].to(device))
        probabilities.append(logits.softmax(dim=-1).float().cpu())
        labels.append(batch['label'])
        print(f'[test]: it {it}/{len(loader)}', end='\r')
    print()
    return torch.cat(probabilities).numpy(), torch.cat(labels).numpy()


def _per_class(labels, probabilities, classes):
    predictions = probabilities.argmax(axis=1)
    class_ids = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=class_ids, zero_division=0
    )
    rows = []
    for k, name in enumerate(classes):
        positives = labels == k
        # One-vs-rest scores; with two classes they are those of class 1 vs 0.
        has_both = 0 < positives.sum() < len(labels)
        rows.append({
            'class': name,
            'support': int(support[k]),
            'precision': precision[k],
            'recall': recall[k],
            'f1': f1[k],
            'auroc': roc_auc_score(positives, probabilities[:, k]) if has_both else np.nan,
            'average_precision': average_precision_score(positives, probabilities[:, k]) if has_both else np.nan,
        })
    table = pd.DataFrame(rows)
    macro = table.drop(columns=['class', 'support']).mean().to_dict()
    table.loc[len(table)] = {'class': 'macro', 'support': int(support.sum()), **macro}
    return table


def _per_dataset(predictions_table, classes):
    rows = []
    for source, group in predictions_table.groupby('source', sort=False):
        for name in classes:
            members = group[group['label'] == name]
            if len(members):
                rows.append({
                    'source': source,
                    'class': name,
                    'support': len(members),
                    'recall': (members['prediction'] == name).mean(),
                })
    return pd.DataFrame(rows)


def _plot_confusion(matrix, classes, title, path):
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)

    size = max(4.0, 1.0 + 0.75 * len(classes))
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    image = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Proporção da classe real')
    colorbar.ax.yaxis.set_major_formatter(lambda value, _: f'{value:.1f}'.replace('.', ','))

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = 'white' if normalized[i, j] > 0.55 else '#1f1f1f'
            ax.text(j, i, f'{normalized[i, j]:.2f}'.replace('.', ',') + f'\n({matrix[i, j]})',
                    ha='center', va='center', fontsize=8, color=color)

    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel('Classe predita')
    ax.set_ylabel('Classe real')
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(schema='bethesda', condition=None):
    configs = yaml.safe_load((Path(__file__).with_name('configs.yaml')).read_text())
    condition = condition or str(configs['DOWNSTREAM']['FROM_PRETRAINED'])
    data_root = PROJECT_ROOT / configs['PROJECT_DATA']
    log_root = PROJECT_ROOT / configs['PROJECT_LOG']
    log_root.mkdir(parents=True, exist_ok=True)
    run_dir = PROJECT_ROOT / configs['PATH_FINETUNED'] / schema / condition
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_dataset, classes = _datasets(
        schema, data_root, configs, EvaluationTransform(),
        float(configs['DATA']['TEST_SIZE']), float(configs['PRETRAINING']['VALIDATION_SIZE'])
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(configs['DOWNSTREAM']['BATCH_SIZE']),
        shuffle=False,
        num_workers=int(configs['DATA']['NUM_WORKERS']),
        pin_memory=torch.cuda.is_available(),
    )
    sources = _sample_sources(test_dataset)

    # The checkpoint holds the whole classifier (ViT, LoRA and head), so the
    # architecture is built from the base ViT and every weight is overwritten.
    lora_args = configs['PRETRAINING']['LORA']
    classifier = LoRAViTClassifier(
        LoRAViTModel(r=lora_args.get('R', 8), alpha=lora_args.get('ALPHA', 16),
                     dropout=lora_args.get('DROPOUT', 0.0), device=device),
        classes, device=device,
    )

    for phase in PHASES:
        checkpoint_path = run_dir / f'vit_classifier_{phase}.pt'
        if not checkpoint_path.exists():
            logger.warning(f'Skipping {phase}: {checkpoint_path} not found')
            continue

        checkpoint = load_checkpoint(classifier, checkpoint_path, device)
        probabilities, labels = _predict(classifier, test_loader, device)

        prefix = log_root / f'eval_{schema}_{condition}_{phase}'
        predictions_table = pd.DataFrame({
            'path': [path for path, _ in sources],
            'source': [source for _, source in sources],
            'label': [classes[k] for k in labels],
            'prediction': [classes[k] for k in probabilities.argmax(axis=1)],
            **{f'prob_{name}': probabilities[:, k] for k, name in enumerate(classes)},
        })
        predictions_table.to_csv(f'{prefix}_predictions.csv', index=False)

        per_class = _per_class(labels, probabilities, classes)
        per_class.to_csv(f'{prefix}_per_class.csv', index=False)
        _per_dataset(predictions_table, classes).to_csv(f'{prefix}_per_dataset.csv', index=False)

        matrix = confusion_matrix(labels, probabilities.argmax(axis=1), labels=np.arange(len(classes)))
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(f'{prefix}_confusion.csv')
        _plot_confusion(matrix, classes, f'{schema} · {condition} · {phase}', f'{prefix}_confusion.png')

        # Macro means over classes, comparable with downstream_test_*.csv
        # (torchmetrics computes the same macro averages).
        macro = per_class.set_index('class').loc['macro']
        logger.info(
            f'[eval] {schema}/{condition}/{phase} (best epoch {checkpoint["epoch"]}): '
            f'recall={macro["recall"]:.4f} auroc={macro["auroc"]:.4f} ap={macro["average_precision"]:.4f}'
        )
        print(per_class.to_string(index=False, float_format='%.3f'))


if __name__ == '__main__':
    main()
