#!/usr/bin/env python3
"""Launch ResNet34/50 QAT; promote validation-selected weights after full test."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
COHORT = REPO / 'runs/resnet34_50_qat_retrain_20261009'
PLANS = {
    'resnet34_snn_v7_phase6': {'gpu': 0, 'batch': 64, 'eval_batch': 64},
    'resnet50_snn_v7_phase6': {'gpu': 3, 'batch': 24, 'eval_batch': 24},
}
SOURCE_FILES = [
    'MNSIM/Interface/network.py',
    'MNSIM/Interface/finetune_snn_v7.py',
    'MNSIM/Interface/train_snn_v7.py',
    'MNSIM/Interface/snn_v7_training_data.py',
    'scripts/finetune_snn_v7.py',
    'scripts/run_resnet34_50_qat_20261009.py',
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        temporary = Path(stream.name)
    os.replace(temporary, path)


def run_directory(model: str) -> Path:
    suffix = '_eval24' if model.startswith('resnet50') else ''
    return REPO / f'runs/{model}_qat_retrain_20261009_b{PLANS[model]["batch"]}_tau2_th05{suffix}'


def launch(models=None) -> None:
    models = list(PLANS) if models is None else list(models)
    for model in models:
        run = run_directory(model)
        if any((run / name).exists() for name in ('controller.log', 'train.log')):
            raise FileExistsError(f'refusing to relaunch existing run: {run}')
        checkpoint = REPO / f'inputs/weights/pretrain/{model}_software_pretrain_best.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    COHORT.mkdir(parents=True, exist_ok=True)
    manifest = COHORT / 'launch.json'
    jobs = json.loads(manifest.read_text()) if manifest.exists() else {}
    for model in models:
        plan = PLANS[model]
        run = run_directory(model)
        run.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in SOURCE_FILES:
            destination = run / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, destination)
            hashes[name] = digest(destination)
        write_json(run / 'source_sha256.json', hashes)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--model', model]
        with (run / 'controller.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (run / 'controller.pid').write_text(f'{process.pid}\n')
        jobs[model] = {**plan, 'controller_pid': process.pid, 'run_dir': str(run)}
        print(f'{model}: GPU {plan["gpu"]}, batch {plan["batch"]}, '
              f'controller PID {process.pid}, log {run / "train.log"}', flush=True)
    write_json(COHORT / 'launch.json', jobs)


def promote(model: str, run: Path, report: dict) -> dict:
    if report['test']['limited_batches'] is not None:
        raise RuntimeError('acceptance requires the full held-out test split')
    config = report['run_config']
    if config['max_train_batches'] is not None or config['max_eval_batches'] is not None:
        raise RuntimeError('acceptance requires unlimited training/evaluation batches')
    if any(config['dataset_splits'][name]['sample_count'] != 4000
           for name in ('finetune', 'calibration', 'test')):
        raise RuntimeError('unexpected V7 dataset split size')
    acceptance = {'passed': report['test']['accuracy'] > 0.90,
                  'model': model, 'best_epoch': report['best_epoch'],
                  'completed_epochs': report['completed_epochs'],
                  'validation_accuracy': report['best_validation_accuracy'],
                  'test_accuracy': report['test']['accuracy'], 'test_samples': 4000,
                  'readout': 'temporal classifier current mean (terminal LIF bypass)',
                  'software_qat_only': True, 'hspice_executed': False}
    if not acceptance['passed']:
        return acceptance
    formal = REPO / 'inputs/weights/finetune'
    formal.mkdir(parents=True, exist_ok=True)
    best = run / f'{model}_software_qat_finetune_best.pth'
    destination = formal / best.name
    history = run / f'{model}_software_qat_finetune_history.json'
    # Serialize both promotions so the checksum manifest includes both models.
    with (formal.parent / '.qat-promotion.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            shutil.copy2(destination, run / 'previous_formal_best.pth')
        with tempfile.NamedTemporaryFile(dir=formal, delete=False) as stream:
            temporary = Path(stream.name)
        shutil.copy2(best, temporary)
        os.replace(temporary, destination)
        shutil.copy2(history, formal / history.name)
        sums = {str(path.relative_to(formal.parent)): digest(path)
                for path in sorted(formal.parent.rglob('*.pth'))}
        with tempfile.NamedTemporaryFile(mode='w', dir=formal.parent, delete=False) as stream:
            stream.write(''.join(f'{value}  {name}\n' for name, value in sorted(sums.items())))
            temporary = Path(stream.name)
        os.replace(temporary, formal.parent / 'SHA256SUMS')
        acceptance.update(formal_best_checkpoint=str(destination), sha256=digest(destination))
    return acceptance


def train(model: str) -> None:
    plan = PLANS[model]
    run = run_directory(model)
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = REPO / f'inputs/weights/pretrain/{model}_software_pretrain_best.pth'
    calibration_batches = max(16, math.ceil(2048 / plan['batch']))
    command = [
        sys.executable, '-u', 'scripts/finetune_snn_v7.py', '--model', model,
        '--checkpoint', str(checkpoint), '--dataset-root',
        'inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6',
        '--sim-config', 'SimConfig.ini', '--output-dir', str(run), '--device', 'cuda:0',
        '--epochs', '20', '--train-batch', str(plan['batch']), '--test-batch', str(plan['eval_batch']),
        '--learning-rate', '0.0001', '--momentum', '0.9', '--weight-decay', '0.00001',
        '--milestones', '8', '14', '--gamma', '0.1', '--gradient-clip-value', '0.5',
        '--lif-tau', '2.0', '--lif-threshold', '0.5', '--readout', 'membrane',
        '--freeze-bn', '--calibration-batches', str(calibration_batches),
        '--evaluate-initial', '--target-validation-accuracy', '0.95',
        '--min-epochs', '1', '--seed', '20261009',
    ]
    env = dict(os.environ)
    overrides = {'CUDA_VISIBLE_DEVICES': str(plan['gpu']), 'OMP_NUM_THREADS': '4',
                 'MNSIM_DVS_NUM_WORKERS': '8', 'MNSIM_SNN_LIF_V_RESET': '0.0',
                 'MNSIM_SNN_LIF_ALPHA': '2.0'}
    env.update(overrides)
    with (run / 'train.log').open('x') as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
        (run / 'train.pid').write_text(f'{process.pid}\n')
        write_json(run / 'launch.json', {'controller_pid': os.getpid(),
                   'training_pid': process.pid, 'command': command, 'cwd': str(REPO),
                   'physical_gpu': plan['gpu'], 'environment': overrides,
                   'source_checkpoint_sha256': digest(checkpoint),
                   'acceptance': 'full held-out test accuracy > 0.90; validation selects checkpoint'})
        print(f'{model} training PID {process.pid}; log: {run / "train.log"}', flush=True)
        exit_code = process.wait()
    if exit_code:
        write_json(run / 'acceptance.json', {'passed': False, 'model': model,
                                           'training_exit_code': exit_code})
        raise SystemExit(f'{model} training failed with exit code {exit_code}')
    report = json.loads((run / f'{model}_software_qat_finetune_history.json').read_text())
    acceptance = promote(model, run, report)
    write_json(run / 'acceptance.json', acceptance)
    print(json.dumps(acceptance, indent=2), flush=True)
    if not acceptance['passed']:
        raise SystemExit('test accuracy did not exceed 90%; weight was not promoted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--launch', action='store_true')
    action.add_argument('--launch-model', choices=sorted(PLANS))
    action.add_argument('--model', choices=sorted(PLANS))
    args = parser.parse_args()
    if args.launch:
        launch()
    elif args.launch_model:
        launch([args.launch_model])
    else:
        train(args.model)
