"""Fresh v25 purple-spinner campaign; never load a v23/v24 policy.

Cache all sensory features again, including spinner clips in both phases.
The existing song-disjoint normal/slow validation and held-out splits remain fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from malecns_cache_parallel import cache_parallel, missing_indices
from malecns_training import song_key
from neural_campaign import atomic_json
from train_v23_full import (BASE, BONUS, LOW, calibrate, clips, entries,
                            load_best, objective, read, train_epoch)
from train_v23_spinner_finetune import selected_long_clips
from warm_validation import WarmValidationEngine, contextual_probes


ROOT = Path('runs/malecns_v25_full')
FEATURE_WIDTH = 13476
PHASE_EPOCHS = {'A': 12, 'C': 8}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    config = read(Path('runs/malecns_v23_full/config.json'))
    config.update(architecture='malecns-image-motor-v25-full-purple-spinner',
                  observation_style='web-native-purple-spinner',
                  phase_plan='A:100 base+22 bonus+30 low-SV; C:50; 2 circle clips/map plus native spinner clips',
                  cache_workers=2)
    config['source_hashes'] = {name: sha(name) for name in (
        'malecns_training.py', 'sensory_readout.py', 'sensory_temporal_policy.py',
        'highres_taiko.py', 'malecns_cache_parallel.py', 'warm_validation.py',
        'train_v23_full.py', 'train_v23_spinner_finetune.py', 'train_v25_full.py')}
    split = read(BASE / 'split.json')
    bonus = read(BONUS / 'rows.json')
    augmented = read(LOW / 'rows.json')
    slow_val = read(LOW / 'validation_rows.json')
    slow_test = read(LOW / 'test_rows.json')
    if tuple(map(len, (split['A'], split['C'], bonus, augmented, slow_val, slow_test))) != (100, 50, 22, 30, 8, 8):
        raise ValueError('Training split changed')
    train_a = split['A'] + bonus + augmented
    train_c = split['C']
    train_songs = {song_key(r) for r in train_a + train_c}
    val_songs = {song_key(r) for r in split['validation'] + slow_val}
    test_songs = {song_key(r) for r in split['test'] + slow_test}
    if train_songs & (val_songs | test_songs) or val_songs & test_songs:
        raise ValueError('Train/validation/test song leakage')
    rows_a = clips(train_a) + selected_long_clips(train_a)
    rows_c = clips(train_c) + selected_long_clips(train_c)
    normal_short = contextual_probes({'validation': split['validation'][:4]}, prelude_ms=2000)
    slow_short = contextual_probes({'validation': slow_val[:4]}, prelude_ms=2000)
    normal_long = selected_long_clips(split['validation'][:12], maximum=1)
    slow_long = selected_long_clips(slow_val, maximum=1)
    if not normal_long or not slow_long:
        raise ValueError('Spinner validation empty')
    if train_songs & {song_key(r) for r in normal_long + slow_long}:
        raise ValueError('Spinner validation song leakage')
    probes = {
        'short_normal': normal_short, 'short_slow': slow_short,
        'spinner_normal': normal_long, 'spinner_slow': slow_long,
        'full_normal': contextual_probes(split, prelude_ms=2000),
        'full_slow': contextual_probes({'validation': slow_val}, prelude_ms=2000),
        'held_normal': contextual_probes({'validation': split['test']}, prelude_ms=2000),
        'held_slow': contextual_probes({'validation': slow_test}, prelude_ms=2000),
    }
    return config, rows_a, rows_c, probes


def reports(engine, policy, probes, prefix='short'):
    return {
        'normal': engine.evaluate(probes[f'{prefix}_normal'], policy),
        'slow': engine.evaluate(probes[f'{prefix}_slow'], policy),
        'spinner_normal': engine.evaluate(probes['spinner_normal'], policy),
        'spinner_slow': engine.evaluate(probes['spinner_slow'], policy),
    }


def quality(result):
    spinner = .5 * (result['spinner_normal']['swell']['coverage'] +
                    result['spinner_slow']['swell']['coverage'])
    return objective(result['normal'], result['slow']) + .2 * spinner


def train_stage(stage, paths, config, engine, probes, initial):
    from sensory_temporal_policy import SensoryTemporalPolicy
    out = ROOT / ('phase_' + stage)
    out.mkdir(exist_ok=True)
    width = np.load(paths[0].with_suffix('.npy'), mmap_mode='r').shape[1]
    policy = SensoryTemporalPolicy(width).cuda()
    if initial is None:
        calibrate(policy, paths)
    else:
        payload = torch.load(initial, map_location='cuda', weights_only=False)
        if payload['feature_width'] != width:
            raise ValueError('Phase sensory width mismatch')
        policy.load_state_dict(payload['policy'])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-4 if stage == 'A' else 8e-5,
                                  weight_decay=.001)
    weights = torch.tensor(config['class_weights'], dtype=torch.float32, device='cuda')
    best, stale, start = -1e9, 0, 1
    last = out / 'last.pt'
    if last.exists():
        payload = torch.load(last, map_location='cuda', weights_only=False)
        if payload['config'] != config:
            raise ValueError('Cannot resume with a changed v25 config')
        policy.load_state_dict(payload['policy'])
        optimizer.load_state_dict(payload['optimizer'])
        best, stale, start = payload['best_objective'], payload['stale'], payload['epoch'] + 1
    for epoch in range(start, PHASE_EPOCHS[stage] + 1):
        loss = train_epoch(policy, optimizer, paths, weights)
        policy.eval()
        result = reports(engine, policy, probes)
        value = quality(result)
        improved = value > best + 1e-5
        best, stale = (value, 0) if improved else (best, stale + 1)
        atomic_json(out / f'epoch-{epoch}.json',
                    {'phase': stage, 'epoch': epoch, 'loss': loss, 'objective': value,
                     'best': improved, **result})
        payload = {'architecture': config['architecture'], 'config': config,
                   'epoch': epoch, 'threshold': .65, 'feature_width': width,
                   'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                   'best_objective': best, 'stale': stale}
        torch.save(payload, last)
        if improved:
            torch.save(payload, out / 'best.pt')
        print(json.dumps({'event': 'v25_epoch', 'phase': stage, 'epoch': epoch,
                          'loss': loss, 'normal_accuracy': result['normal']['accuracy'],
                          'slow_accuracy': result['slow']['accuracy'],
                          'normal_miss': result['normal']['miss'],
                          'slow_miss': result['slow']['miss'],
                          'normal_false_hits': result['normal']['false_hits'],
                          'slow_false_hits': result['slow']['false_hits'],
                          'normal_spinner_coverage': result['spinner_normal']['swell']['coverage'],
                          'slow_spinner_coverage': result['spinner_slow']['swell']['coverage'],
                          'best': improved}), flush=True)
        if epoch >= 4 and stale >= 4:
            print(json.dumps({'event': 'v25_early_stop', 'phase': stage, 'epoch': epoch}), flush=True)
            break
    if not (out / 'best.pt').exists():
        raise RuntimeError('No best v25 checkpoint for phase ' + stage)
    return out / 'best.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(45)
    np.random.seed(45)
    random.seed(45)
    config, rows_a, rows_c, probes = prepare()
    ROOT.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'config.json'
    if config_path.exists() and read(config_path) != config:
        raise ValueError('v25 source/config changed; choose a new run root')
    atomic_json(config_path, config)
    for path, rows in ((ROOT / 'rows_A.json', rows_a), (ROOT / 'rows_C.json', rows_c)):
        if path.exists() and read(path) != rows:
            raise ValueError('v25 training clips changed')
        atomic_json(path, rows)
    estimate = sum(int(r['excerpt_length_ms'] / 8) * FEATURE_WIDTH * 4
                   for r in rows_a + rows_c) / 1024**3
    print(json.dumps({'event': 'v25_preflight', 'A_clips': len(rows_a),
                      'C_clips': len(rows_c), 'spinner_normal_probes': len(probes['spinner_normal']),
                      'spinner_slow_probes': len(probes['spinner_slow']),
                      'estimated_cache_gib': estimate,
                      'free_gib': shutil.disk_usage(ROOT).free / 1024**3}), flush=True)
    if args.preflight_only:
        return
    if shutil.disk_usage(ROOT).free < (estimate + 18) * 1024**3:
        raise RuntimeError('Disk reserve below estimated fresh cache + 18 GiB')
    engine = WarmValidationEngine(config, ROOT)
    cache_a, cache_c = ROOT / 'cache_A', ROOT / 'cache_C'
    cache_parallel(engine, rows_a, cache_a, workers=2, stage='A')
    if missing_indices(rows_a, cache_a, config, 'clean'):
        raise RuntimeError('Incomplete v25 A cache')
    if args.cache_only:
        cache_parallel(engine, rows_c, cache_c, workers=2, stage='C')
        return
    best_a = train_stage('A', entries(cache_a), config, engine, probes, None)
    policy_a, _ = load_best(best_a)
    a_report = reports(engine, policy_a, probes, prefix='full')
    atomic_json(ROOT / 'A_validation.json', a_report)
    del policy_a
    print(json.dumps({'event': 'v25_A_validation', 'normal_accuracy': a_report['normal']['accuracy'],
                      'slow_accuracy': a_report['slow']['accuracy']}), flush=True)
    cache_parallel(engine, rows_c, cache_c, workers=2, stage='C')
    if missing_indices(rows_c, cache_c, config, 'clean'):
        raise RuntimeError('Incomplete v25 C cache')
    best_c = train_stage('C', entries(cache_c), config, engine, probes, best_a)
    policy_c, _ = load_best(best_c)
    c_report = reports(engine, policy_c, probes, prefix='full')
    atomic_json(ROOT / 'C_validation.json', c_report)
    selected = best_c if quality(c_report) > quality(a_report) else best_a
    policy, payload = load_best(selected)
    held = {'normal': engine.evaluate(probes['held_normal'], policy),
            'slow': engine.evaluate(probes['held_slow'], policy)}
    result = {'status': 'completed', 'selected_checkpoint': str(selected),
              'selected_phase': 'C' if selected == best_c else 'A', 'epoch': payload['epoch'],
              'A_validation': a_report, 'C_validation': c_report, 'heldout': held,
              'comparison_only': {'v23': 'runs/malecns_v23_full/result.json',
                                  'v24': 'runs/malecns_v24_purple_spinner_finetune/result.json'}}
    atomic_json(ROOT / 'result.json', result)
    print(json.dumps({'event': 'v25_complete', 'selected': str(selected),
                      'heldout_normal_accuracy': held['normal']['accuracy'],
                      'heldout_slow_accuracy': held['slow']['accuracy']}), flush=True)


if __name__ == '__main__':
    main()
