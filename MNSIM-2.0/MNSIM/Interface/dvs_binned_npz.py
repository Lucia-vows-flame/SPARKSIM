#-*-coding:utf-8-*-
"""按样本 `.npz` 文件组织的预分箱 DVS 数据集加载器。

这里落实的是本项目已经确认的数据契约：
- 一个 `.npz` 对应一个样本；
- 脉冲张量形状为 `[T, C, H, W]`；
- 通道数量与语义由数据集 `meta.json` 声明；
- 进入模型前会裁成二值脉冲。
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from MNSIM.Accuracy_Model.snn_accuracy_contract import canonical_json_sha256

NUM_CLASSES = int(os.environ.get('MNSIM_DVS_NUM_CLASSES', '10'))

TRAIN_BATCH_SIZE = int(os.environ.get('MNSIM_DVS_TRAIN_BATCH', '64'))
TEST_BATCH_SIZE = int(os.environ.get('MNSIM_DVS_TEST_BATCH', '64'))
NUM_WORKERS = int(os.environ.get('MNSIM_DVS_NUM_WORKERS', '0'))
TRAIN_SPLIT = os.environ.get('MNSIM_DVS_TRAIN_SPLIT', 'train')
TEST_SPLIT = os.environ.get('MNSIM_DVS_TEST_SPLIT', 'test')

DEFAULT_ROOT = os.environ.get(
    'MNSIM_DVS_BINNED_ROOT',
    os.path.join(os.path.dirname(__file__), 'dvs_binned'),
)

TARGET_H = int(os.environ.get('MNSIM_DVS_H', '32'))
TARGET_W = int(os.environ.get('MNSIM_DVS_W', '32'))


def _optional_positive_int_env(name: str):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


TEST_SAMPLE_LIMIT = _optional_positive_int_env("MNSIM_DVS_TEST_SAMPLE_LIMIT")
TEST_SAMPLES_PER_CLASS = _optional_positive_int_env("MNSIM_DVS_TEST_SAMPLES_PER_CLASS")


@dataclass(frozen=True)
class DVSMeta:
    """从 `meta.json` 中读取的可选元数据。"""

    t_bins: Optional[int] = None
    h: Optional[int] = None
    w: Optional[int] = None
    channels: Optional[int] = None
    num_classes: Optional[int] = None


def _load_meta(root: str) -> DVSMeta:
    """当数据集制作者提供 `meta.json` 时，读取数据集级元数据。"""
    meta_path = os.path.join(root, 'meta.json')
    if not os.path.exists(meta_path):
        return DVSMeta()
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return DVSMeta(
        t_bins=meta.get('T_bins', meta.get('t_bins')),
        h=meta.get('H', meta.get('h')),
        w=meta.get('W', meta.get('w')),
        channels=meta.get('C', meta.get('c')),
        num_classes=meta.get('num_classes', meta.get('NUM_CLASSES')),
    )


# 提前加载 `meta.json`，这样即使调用方先建网络再取 dataloader，
# 也能看到正确的 `NUM_CLASSES`。
_IMPORT_META = _load_meta(DEFAULT_ROOT)
if _IMPORT_META.num_classes is not None:
    NUM_CLASSES = int(_IMPORT_META.num_classes)


def _list_npz(dir_path: str):
    """返回某个 split 下排序后的文件列表。"""
    if not os.path.isdir(dir_path):
        return []
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.npz')]
    files.sort()
    return files


def _resolve_split_dir(root: str, split: str) -> tuple[str, str]:
    """把逻辑 split 名解析到磁盘上的真实目录。"""

    split = str(split)
    candidate_map = {
        'train': ['train', 'pretrain', 'finetune'],
        'test': ['test'],
        'pretrain': ['pretrain'],
        'finetune': ['finetune'],
        'calibration': ['calibration'],
    }
    candidates = candidate_map.get(split, [split])
    for candidate in candidates:
        split_dir = os.path.join(root, candidate)
        if os.path.isdir(split_dir):
            return candidate, split_dir
    raise FileNotFoundError(
        f'cannot resolve split={split!r} under {root}; tried {candidates}'
    )


def _read_npz(path: str):
    """读取一个样本，同时兼容当前与历史字段名。"""
    with np.load(path, allow_pickle=False) as data:
        if 'x_seq' in data:
            x = data['x_seq']
        elif 'x' in data:
            x = data['x']
        else:
            raise KeyError(f'{path}: missing x_seq/x')
        if 'y' in data:
            y = int(data['y'])
        elif 'label' in data:
            y = int(data['label'])
        else:
            raise KeyError(f'{path}: missing y/label')
    return x, y


def _to_seq_tensor(
    x_np: np.ndarray, expected_channels: Optional[int] = None
) -> torch.Tensor:
    """把原始 numpy 样本转换为模型要求的张量契约。"""
    if x_np.ndim != 4:
        raise ValueError(f'expected x_seq [T,C,H,W], got {x_np.shape}')
    if expected_channels is not None and x_np.shape[1] != int(expected_channels):
        raise ValueError(
            f'expected C={int(expected_channels)} from dataset metadata, got {x_np.shape}'
        )
    x = torch.from_numpy(x_np)
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    x = (x > 0).to(torch.float32)
    return x


def _resize_seq(x_seq: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Use binary spatial OR to reduce `[T,C,H,W]` to the model size."""
    t, c, h, w = x_seq.shape
    target_h, target_w = int(hw[0]), int(hw[1])
    if (h, w) == (target_h, target_w):
        return x_seq
    x = x_seq.reshape(t * c, 1, h, w)
    x = F.adaptive_max_pool2d(x, output_size=(target_h, target_w))
    return x.reshape(t, c, target_h, target_w)


class DVSBinnedNPZDataset(Data.Dataset):
    """对目录中逐样本 `.npz` 文件做的一层简单封装。"""

    def __init__(self, root: str, split: str, target_hw: Tuple[int, int] = (32, 32)):
        super().__init__()
        self.root = root
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))
        self.expected_channels = _load_meta(root).channels or 2

        requested_split = str(split)
        resolved_split, split_dir = _resolve_split_dir(root, requested_split)
        self.split = requested_split
        self.resolved_split = resolved_split
        self.split_dir = Path(split_dir)
        self.files = _list_npz(split_dir)
        if len(self.files) == 0:
            raise FileNotFoundError(
                f'No .npz files found under {split_dir}. '
                f'Expected per-sample npz with keys x_seq([T,C,H,W]) and y/label.'
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x_np, y = _read_npz(self.files[int(idx)])
        x_seq = _to_seq_tensor(x_np, expected_channels=self.expected_channels)
        x_seq = _resize_seq(x_seq, self.target_hw)
        return x_seq, y

    def accuracy_split_identity(self):
        entries = [
            {
                "relative_path": path.relative_to(self.split_dir).as_posix(),
                "size_bytes": path.stat().st_size,
            }
            for path in map(Path, self.files)
        ]
        return {
            "dataset": f"{type(self).__module__}.{type(self).__qualname__}",
            "root": str(Path(self.root).resolve()),
            "split": self.split,
            "resolved_split": self.resolved_split,
            "sample_count": len(entries),
            "sample_manifest_sha256": canonical_json_sha256(entries),
        }


def get_calibration_dataloader(root: str = DEFAULT_ROOT):
    """Build only the training loader used by formal accuracy calibration."""
    meta = _load_meta(root)
    if meta.num_classes is not None:
        global NUM_CLASSES
        NUM_CLASSES = int(meta.num_classes)
    train_dataset = DVSBinnedNPZDataset(
        root=root,
        split=TRAIN_SPLIT,
        target_hw=(TARGET_H, TARGET_W),
    )
    return Data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )


def get_dataloader(root: str = DEFAULT_ROOT):
    """为预分箱 DVS 数据集构建 train/test dataloader。"""
    meta = _load_meta(root)
    if meta.num_classes is not None:
        global NUM_CLASSES
        NUM_CLASSES = int(meta.num_classes)

    train_dataset = DVSBinnedNPZDataset(root=root, split=TRAIN_SPLIT, target_hw=(TARGET_H, TARGET_W))
    test_dataset = DVSBinnedNPZDataset(root=root, split=TEST_SPLIT, target_hw=(TARGET_H, TARGET_W))
    if TEST_SAMPLE_LIMIT is not None:
        if TEST_SAMPLE_LIMIT > len(test_dataset.files):
            raise ValueError(
                f"MNSIM_DVS_TEST_SAMPLE_LIMIT={TEST_SAMPLE_LIMIT} exceeds available test samples {len(test_dataset.files)}"
            )
        test_dataset.files = test_dataset.files[:TEST_SAMPLE_LIMIT]
    if TEST_SAMPLES_PER_CLASS is not None:
        selected = []
        counts = {}
        for path in test_dataset.files:
            _, label = _read_npz(path)
            if counts.get(label, 0) < TEST_SAMPLES_PER_CLASS:
                selected.append(path)
                counts[label] = counts.get(label, 0) + 1
        if len(counts) < NUM_CLASSES or any(v < TEST_SAMPLES_PER_CLASS for v in counts.values()):
            raise ValueError('not enough test samples for requested per-class limit')
        test_dataset.files = selected

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, test_loader
