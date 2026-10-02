"""Shared sensory-memory campaign helpers used by the current v25 trainer.

Two 8-second clips per source chart keep float32 neural caches within disk
budget. Chart data supplies only expert labels; inference remains visual.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from training.common.cache_parallel import cache_parallel, missing_indices
from flytaiko.malecns_training import excerpt, score, song_key
from flytaiko.neural_campaign import atomic_json
from flytaiko.sensory_temporal_policy import SensoryTemporalPolicy
from flytaiko.warm_validation import WarmValidationEngine, contextual_probes


ROOT = Path('runs/malecns_v23_full')
BASE = Path('runs/malecns_v18_osu_sv_campaign')
BONUS = Path('runs/malecns_v18_bonus_cache')
LOW = Path('runs/malecns_v20_low_sv_augmentation')
PILOT = Path('runs/malecns_v23_sensory_memory_pilot')
CIRCLE = {'don', 'kat', 'don_big', 'kat_big'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clips(rows):
    result = []
    for row in rows:
        notes = [n for n in row['notes'] if n['type'] in CIRCLE]
        if len(notes) < 4:
            raise ValueError('Too few circle notes: ' + row['source'])
        # One regular interval and one locally crowded interval per chart.
        regular = notes[len(notes) // 3]['t']
        pairs = [(abs((b['t'] - a['t']) * a['scroll_px_per_ms']), a['t'])
                 for a, b in zip(notes, notes[1:]) if a['t'] > 10000]
        crowded = min(pairs)[1] if pairs else notes[2 * len(notes) // 3]['t']
        for centre in (regular, crowded):
            piece = excerpt(row, centre, 8000)
            if piece is None:
                raise ValueError('Cannot extract clip: ' + row['source'])
            result.append(piece)
    return result


def objective(normal, slow):
    def value(report):
        n = max(1, report['total_notes'])
        return score(report) - .4 * (report['miss'] + report['false_hits']) / n
    return .5 * (value(normal) + value(slow))


def prepare():
    config = read(PILOT / 'config.json')
    config.update(architecture='malecns-image-motor-v23-full-sensory-memory',
                  phase_plan='A:100 base+22 bonus+30 low-SV augment; C:50; two 8s clips/map')
    config['source_hashes'] = {name: digest(name) for name in (
        'flytaiko/malecns_training.py', 'flytaiko/sensory_readout.py', 'flytaiko/sensory_temporal_policy.py',
        'flytaiko/highres_taiko.py', 'training/common/cache_parallel.py',
        'flytaiko/warm_validation.py', 'training/v23/train_full.py')}
    split = read(BASE / 'split.json')
    bonus = read(BONUS / 'rows.json')
    augmented = read(LOW / 'rows.json')
    slow_val = read(LOW / 'validation_rows.json')
    slow_test = read(LOW / 'test_rows.json')
    if tuple(map(len, (split['A'], split['C'], bonus, augmented, slow_val, slow_test))) != (100, 50, 22, 30, 8, 8):
        raise ValueError('Full campaign source split changed')
    train = split['A'] + bonus + augmented + split['C']
    validation = split['validation'] + slow_val
    test = split['test'] + slow_test
    if ({song_key(r) for r in train} & {song_key(r) for r in validation + test} or
            {song_key(r) for r in validation} & {song_key(r) for r in test}):
        raise ValueError('Train/validation/test song leakage')
    rows_a = clips(split['A'] + bonus + augmented)
    rows_c = clips(split['C'])
    if len(rows_a) != 304 or len(rows_c) != 100:
        raise ValueError('Clip count changed')
    normal = contextual_probes({'validation': split['validation'][:4]}, prelude_ms=2000)
    slow = contextual_probes({'validation': slow_val[:4]}, prelude_ms=2000)
    full_normal = contextual_probes(split, prelude_ms=2000)
    full_slow = contextual_probes({'validation': slow_val}, prelude_ms=2000)
    held_normal = contextual_probes({'validation': split['test']}, prelude_ms=2000)
    held_slow = contextual_probes({'validation': slow_test}, prelude_ms=2000)
    return config, rows_a, rows_c, normal, slow, full_normal, full_slow, held_normal, held_slow


def entries(directory):
    paths = sorted(directory.glob('*.json'))
    if not paths:
        raise ValueError('Empty cache: ' + str(directory))
    return paths


def calibrate(policy, paths):
    examples = []
    for path in paths:
        data = np.load(path.with_suffix('.npy'), mmap_mode='r')
        frames = read(path)['frames']
        indices = np.linspace(0, frames - 1, min(64, frames), dtype=int)
        examples.append(np.asarray(data[indices], dtype=np.float32))
    policy.calibrate(np.concatenate(examples))


def train_epoch(policy, optimizer, paths, weights):
    policy.train()
    order = list(paths)
    random.shuffle(order)
    losses = []
    for path in order:
        data = np.load(path.with_suffix('.npy'), mmap_mode='r')
        labels = np.load(path.with_name(path.stem + '-labels.npy')).astype(np.int64)
        frames = read(path)['frames']
        if frames != len(labels) or len(data) < frames:
            raise RuntimeError('Cache length mismatch: ' + str(path))
        state = None
        for start in range(0, frames, 128):
            end = min(frames, start + 128)
            x = torch.as_tensor(np.asarray(data[start:end], dtype=np.float32).copy(), device='cuda')[None]
            y = torch.as_tensor(labels[start:end], device='cuda')
            logits, state = policy.sequence(x, state)
            loss = torch.nn.functional.cross_entropy(logits[0], y, weight=weights)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite v23 training loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5)
            optimizer.step()
            state = state.detach()
            losses.append(float(loss))
    return float(np.mean(losses))


def train_stage(stage, paths, config, engine, normal, slow, initial, epochs):
    output = ROOT / ('phase_' + stage)
    output.mkdir(exist_ok=True)
    width = np.load(paths[0].with_suffix('.npy'), mmap_mode='r').shape[1]
    policy = SensoryTemporalPolicy(width).cuda()
    if initial is None:
        calibrate(policy, paths)
    else:
        payload = torch.load(initial, map_location='cuda', weights_only=False)
        if payload['feature_width'] != width:
            raise ValueError('Initial sensory width changed')
        policy.load_state_dict(payload['policy'])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-4 if stage == 'A' else 8e-5,
                                  weight_decay=.001)
    weights = torch.tensor(config['class_weights'], dtype=torch.float32, device='cuda')
    best = -1e9
    stale = 0
    start_epoch = 1
    last = output / 'last.pt'
    if last.exists():
        payload = torch.load(last, map_location='cuda', weights_only=False)
        policy.load_state_dict(payload['policy'])
        optimizer.load_state_dict(payload['optimizer'])
        start_epoch = payload['epoch'] + 1
        best = payload['best_objective']
        stale = payload['stale']
    for epoch in range(start_epoch, epochs + 1):
        loss = train_epoch(policy, optimizer, paths, weights)
        policy.eval()
        normal_report = engine.evaluate(normal, policy, threshold=.65)
        slow_report = engine.evaluate(slow, policy, threshold=.65)
        value = objective(normal_report, slow_report)
        improved = value > best + 1e-5
        if improved:
            best, stale = value, 0
        else:
            stale += 1
        record = {'phase': stage, 'epoch': epoch, 'loss': loss,
                  'normal': normal_report, 'slow': slow_report,
                  'objective': value, 'best': improved}
        atomic_json(output / f'epoch-{epoch}.json', record)
        state = {'architecture': config['architecture'], 'config': config,
                 'epoch': epoch, 'threshold': .65, 'feature_width': width,
                 'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                 'best_objective': best, 'stale': stale}
        torch.save(state, last)
        if improved:
            torch.save(state, output / 'best.pt')
        print(json.dumps({'event': 'v23_epoch', 'phase': stage, 'epoch': epoch,
                          'loss': loss, 'normal_accuracy': normal_report['accuracy'],
                          'slow_accuracy': slow_report['accuracy'],
                          'normal_miss': normal_report['miss'],
                          'slow_miss': slow_report['miss'],
                          'slow_false_hits': slow_report['false_hits'],
                          'best': improved}), flush=True)
        if epoch >= 4 and stale >= 4:
            print(json.dumps({'event': 'v23_early_stop', 'phase': stage, 'epoch': epoch}), flush=True)
            break
    best_path = output / 'best.pt'
    if not best_path.exists():
        raise RuntimeError('No best checkpoint: ' + stage)
    return best_path


def load_best(path):
    payload = torch.load(path, map_location='cuda', weights_only=False)
    policy = SensoryTemporalPolicy(payload['feature_width']).cuda()
    policy.load_state_dict(payload['policy'])
    policy.eval()
    return policy, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    config, rows_a, rows_c, normal, slow, full_normal, full_slow, held_normal, held_slow = prepare()
    ROOT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'config.json'
    if config_path.exists() and read(config_path) != config:
        raise ValueError('Full v23 source/config changed; choose a new run root')
    atomic_json(config_path, config)
    for path, rows in ((ROOT / 'rows_A.json', rows_a), (ROOT / 'rows_C.json', rows_c)):
        if path.exists() and read(path) != rows:
            raise ValueError('Full v23 training clips changed: ' + str(path))
        atomic_json(path, rows)
    estimated_gib = sum(int(r['excerpt_length_ms'] / 8) * 13476 * 4
                        for r in rows_a + rows_c) / 1024**3
    print(json.dumps({'event': 'v23_full_preflight', 'A_clips': len(rows_a),
                      'C_clips': len(rows_c), 'normal_probes': len(full_normal),
                      'slow_probes': len(full_slow), 'estimated_cache_gib': estimated_gib,
                      'free_gib': shutil.disk_usage(ROOT).free / 1024**3}), flush=True)
    if args.preflight_only:
        return
    if shutil.disk_usage(ROOT).free < (estimated_gib + 25) * 1024**3:
        raise RuntimeError('Disk reserve too low for full float32 v23 cache')
    engine = WarmValidationEngine(config, ROOT)
    cache_a, cache_c = ROOT / 'cache_A', ROOT / 'cache_C'
    cache_parallel(engine, rows_a, cache_a, workers=2, stage='A')
    if missing_indices(rows_a, cache_a, config, 'clean'):
        raise RuntimeError('Incomplete v23 A cache')
    if not args.cache_only:
        best_a = train_stage('A', entries(cache_a), config, engine, normal, slow, None, 12)
        policy_a, _ = load_best(best_a)
        a_report = {'normal': engine.evaluate(full_normal, policy_a),
                    'slow': engine.evaluate(full_slow, policy_a)}
        atomic_json(ROOT / 'A_validation.json', a_report)
        print(json.dumps({'event': 'v23_A_validation',
                          'normal_accuracy': a_report['normal']['accuracy'],
                          'slow_accuracy': a_report['slow']['accuracy']}), flush=True)
        del policy_a
    cache_parallel(engine, rows_c, cache_c, workers=2, stage='C')
    if missing_indices(rows_c, cache_c, config, 'clean'):
        raise RuntimeError('Incomplete v23 C cache')
    if args.cache_only:
        return
    best_c = train_stage('C', entries(cache_c), config, engine, normal, slow, best_a, 8)
    policy_c, _ = load_best(best_c)
    c_report = {'normal': engine.evaluate(full_normal, policy_c),
                'slow': engine.evaluate(full_slow, policy_c)}
    atomic_json(ROOT / 'C_validation.json', c_report)
    a_report = read(ROOT / 'A_validation.json')
    select_c = objective(c_report['normal'], c_report['slow']) > objective(a_report['normal'], a_report['slow'])
    selected = best_c if select_c else best_a
    policy, payload = load_best(selected)
    result = {'status': 'completed', 'selected_checkpoint': str(selected),
              'selected_phase': 'C' if select_c else 'A', 'epoch': payload['epoch'],
              'A_validation': a_report, 'C_validation': c_report,
              'heldout_normal': engine.evaluate(held_normal, policy),
              'heldout_slow': engine.evaluate(held_slow, policy)}
    atomic_json(ROOT / 'result.json', result)
    print(json.dumps({'event': 'v23_full_complete', 'selected': str(selected),
                      'heldout_normal_accuracy': result['heldout_normal']['accuracy'],
                      'heldout_slow_accuracy': result['heldout_slow']['accuracy']}), flush=True)


if __name__ == '__main__':
    main()
