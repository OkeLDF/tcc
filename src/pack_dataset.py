"""
Pack the 7 pretraining datasets into a single .zip that preserves the exact
directory layout the loaders in data.py expect, plus a manifest that lets the
remote machine verify the train/valid splits came out identical.

Place this file next to step_pretrain.py (it reads configs.yaml and imports
data.py from the same folder).

Usage:
    python pack_dataset.py pack --dry-run          # inventory only, no zip
    python pack_dataset.py pack --out dataset.zip  # build the archive
    python pack_dataset.py verify                  # run on the pod after extracting

Why ship EVERY image (test partition included):
    BaseCervicalCytologyDataset computes a stratified split over the FULL list
    returned by _collect_samples(). If any image is missing on the pod, the
    split is recomputed over a different set and train/valid change silently,
    breaking the "pretraining and downstream share partitions" contract.
    HiCervix also asserts that train/val/test folders and CSVs all exist.
"""

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import yaml

from data import (
    BTMDataset,
    CPSMI2025Dataset,
    HerlevDataset,
    HiCervixDataset,
    MendeleyLBCDataset,
    PapicitoDataset,
    SIPaKMeDDataset,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # same rule as step_pretrain.py (parents[1])
MANIFEST_NAME = '_pack_manifest.json'

DATASETS = [
    ('PATH_CPSMI2025', CPSMI2025Dataset),
    ('PATH_HERLEV', HerlevDataset),
    ('PATH_MENDELEYLBC', MendeleyLBCDataset),
    ('PATH_SIPAKMED', SIPaKMeDDataset),
    ('PATH_BTM', BTMDataset),
    ('PATH_HICERVIX', HiCervixDataset),
    ('PATH_PAPICITO', PapicitoDataset),
]

# Non-image files the loaders read directly.
AUX_FILES = {
    'PATH_BTM': ['manifest.csv'],
    'PATH_HICERVIX': ['train.csv', 'val.csv', 'test.csv'],
}

# Already-compressed formats: storing is faster and deflate gains ~nothing.
STORED_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def load_settings():
    configs = yaml.safe_load((SCRIPT_DIR / 'configs.yaml').read_text())
    data_root = PROJECT_ROOT / configs['PROJECT_DATA']
    test_size = float(configs['DATA']['TEST_SIZE'])
    validation_size = float(configs['PRETRAINING']['VALIDATION_SIZE'])
    return configs, data_root, test_size, validation_size


def collect_all(cls, path):
    """Every image the loader sees, before any split is applied."""
    obj = cls.__new__(cls)
    obj.path = Path(path)
    obj.task = 'pretraining'
    obj.split = 'train'
    obj.transform = None
    obj.use_first_class_level = False  # CPSMI2025 default
    obj.label_level = 1                # HiCervix default
    return [Path(p) for p, _ in obj._collect_samples()]


def split_fingerprint(cls, path, split, test_size, validation_size, data_root):
    """(count, sha256) of the ordered relative paths in a pretraining split."""
    ds = cls(
        path, task='pretraining', split=split, transform=None,
        test_size=test_size, validation_size=validation_size,
    )
    rel = [Path(p).relative_to(data_root).as_posix() for p in ds.samples[:, 0]]
    digest = hashlib.sha256('\n'.join(rel).encode('utf-8')).hexdigest()
    return len(rel), digest


def human(n_bytes):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n_bytes < 1024:
            return f'{n_bytes:.1f} {unit}'
        n_bytes /= 1024
    return f'{n_bytes:.1f} PB'


def build_manifest(configs, data_root, test_size, validation_size):
    files = set()
    manifest = {'project_data': configs['PROJECT_DATA'], 'datasets': {}}

    for key, cls in DATASETS:
        ds_path = data_root / configs[key]
        images = collect_all(cls, ds_path)

        aux = [ds_path / name for name in AUX_FILES.get(key, [])]
        for f in aux:
            assert f.is_file(), f'Missing auxiliary file {str(f)!r}.'

        entry = {
            'config_key': key,
            'path': configs[key],
            'n_images': len(images),
            'bytes': sum(f.stat().st_size for f in images),
            'by_suffix': {},
        }
        for f in images:
            s = f.suffix.lower()
            entry['by_suffix'][s] = entry['by_suffix'].get(s, 0) + 1

        for split in ('train', 'valid'):
            n, digest = split_fingerprint(cls, ds_path, split, test_size, validation_size, data_root)
            entry[split] = {'n': n, 'sha256': digest}

        manifest['datasets'][cls.__name__] = entry
        files.update(images)
        files.update(aux)

        print(
            f'{cls.__name__:<20} {len(images):>7} imgs  {human(entry["bytes"]):>10}  '
            f'train={entry["train"]["n"]:<6} valid={entry["valid"]["n"]:<6} {entry["by_suffix"]}'
        )

    total_imgs = sum(e['n_images'] for e in manifest['datasets'].values())
    total_bytes = sum(e['bytes'] for e in manifest['datasets'].values())
    print(f'{"TOTAL":<20} {total_imgs:>7} imgs  {human(total_bytes):>10}')
    return manifest, sorted(files)


def pack(args):
    configs, data_root, test_size, validation_size = load_settings()
    print(f'data_root = {data_root}\n')

    manifest, files = build_manifest(configs, data_root, test_size, validation_size)
    if args.dry_run:
        print('\n--dry-run: no archive written.')
        return

    out = Path(args.out).resolve()
    t0 = time.time()
    with zipfile.ZipFile(out, 'w', allowZip64=True) as zf:
        for i, f in enumerate(files, 1):
            arcname = f.relative_to(data_root).as_posix()
            ctype = zipfile.ZIP_STORED if f.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            zf.write(f, arcname, compress_type=ctype, compresslevel=6)
            if i % 500 == 0 or i == len(files):
                print(f'\r[zip] {i}/{len(files)}  {time.time() - t0:.0f}s', end='', flush=True)
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f'\n\nWrote {out}  ({human(out.stat().st_size)})')
    print(f'Extract into: <PROJECT_ROOT>/{configs["PROJECT_DATA"]}')


def verify(args):
    configs, data_root, test_size, validation_size = load_settings()
    manifest_path = Path(args.manifest) if args.manifest else data_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    print(f'data_root = {data_root}\nmanifest  = {manifest_path}\n')

    ok = True
    for key, cls in DATASETS:
        expected = manifest['datasets'][cls.__name__]
        ds_path = data_root / configs[key]
        problems = []

        n_images = len(collect_all(cls, ds_path))
        if n_images != expected['n_images']:
            problems.append(f'n_images {n_images} != {expected["n_images"]}')

        for split in ('train', 'valid'):
            n, digest = split_fingerprint(cls, ds_path, split, test_size, validation_size, data_root)
            if n != expected[split]['n'] or digest != expected[split]['sha256']:
                problems.append(f'{split} split differs (n={n}, expected {expected[split]["n"]})')

        status = 'OK' if not problems else 'FAIL: ' + '; '.join(problems)
        print(f'{cls.__name__:<20} {status}')
        ok &= not problems

    print('\nAll splits identical.' if ok else '\nMISMATCH - do not start training.')
    sys.exit(0 if ok else 1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('pack', help='build dataset.zip')
    p.add_argument('--out', default=str(PROJECT_ROOT / 'dataset.zip'))
    p.add_argument('--dry-run', action='store_true', help='inventory only, no zip')

    v = sub.add_parser('verify', help='check splits after extraction')
    v.add_argument('--manifest', default=None, help=f'default: <data_root>/{MANIFEST_NAME}')

    args = parser.parse_args()
    pack(args) if args.command == 'pack' else verify(args)


if __name__ == '__main__':
    main()
