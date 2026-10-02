"""Fine-tune selected v23 Phase C on native spinner clips, preserving old best.

All optimization clips come from A/C/bonus/augmented training songs. Normal and
slow-SV validation remain disjoint; the original Phase C checkpoint is never
overwritten or promoted automatically.
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
from malecns_training import excerpt
from neural_campaign import atomic_json
from sensory_temporal_policy import SensoryTemporalPolicy
from train_v23_full import (BASE, BONUS, LOW, ROOT as FULL, entries, load_best,
                            objective, read, train_epoch)
from warm_validation import WarmValidationEngine, contextual_probes


ROOT = Path('runs/malecns_v23_spinner_finetune')
INITIAL = FULL / 'phase_C/best.pt'


def selected_long_clips(rows, maximum=2):
    clips = []
    for row in rows:
        longs = [note for note in row['notes'] if note['type'] == 'swell']
        if not longs:
            continue
        chosen = longs if len(longs) <= maximum else [longs[0], longs[-1]]
        for note in chosen:
            length = max(8000, note['end_t'] - note['t'] + 2000)
            clip = excerpt(row, note['t'], length)
            if clip is None or not any(n['type'] == 'swell' and n['t'] == min(1000, note['t']) for n in clip['notes']):
                raise ValueError('Spinner clip extraction lost target object')
            clips.append(clip)
    return clips


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--purple-spinner', action='store_true',
                        help='New purple-spinner sensory cache/run; leave published v23 untouched')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20:
        raise ValueError('epochs must be 1..20')
    if args.purple_spinner:
        ROOT = Path('runs/malecns_v24_purple_spinner_finetune')
    torch.set_num_threads(2)
    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)
    config = read(FULL / 'config.json')
    if args.purple_spinner:
        config['observation_style'] = 'web-native-purple-spinner'
        config['architecture'] = 'malecns-image-motor-v24-purple-spinner-finetune'
        for name in ('highres_taiko.py', 'malecns_training.py'):
            config['source_hashes'][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    split = read(BASE / 'split.json')
    bonus = read(BONUS / 'rows.json')
    augmented = read(LOW / 'rows.json')
    slow_val = read(LOW / 'validation_rows.json')
    rows = selected_long_clips(split['A'] + bonus + augmented + split['C'])
    if len(rows) < 150:
        raise ValueError('Insufficient native spinner training clips')
    normal = contextual_probes({'validation': split['validation'][:4]}, prelude_ms=2000)
    slow = contextual_probes({'validation': slow_val[:4]}, prelude_ms=2000)
    normal_long = selected_long_clips(split['validation'][:12], maximum=1)
    slow_long = selected_long_clips(slow_val, maximum=1)
    if not normal_long or not slow_long:
        raise ValueError('Spinner validation is empty')
    def songs(items):
        return {(r['metadata']['artist'].casefold(), r['metadata']['title'].casefold()) for r in items}
    if songs(rows) & songs(normal + slow + normal_long + slow_long):
        raise ValueError('Train/validation song leakage')
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {'config_sha256': hashlib.sha256((FULL / 'config.json').read_bytes()).hexdigest(),
                'initial_sha256': hashlib.sha256(INITIAL.read_bytes()).hexdigest(),
                'observation_style': config['observation_style'],
                'training_clips': len(rows), 'normal_probes': len(normal),
                'slow_probes': len(slow), 'normal_spinner_probes': len(normal_long),
                'slow_spinner_probes': len(slow_long),
                'source_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (ROOT / 'manifest.json').exists() and read(ROOT / 'manifest.json') != manifest:
        raise ValueError('Spinner fine-tune manifest changed; choose new run root')
    atomic_json(ROOT / 'manifest.json', manifest)
    atomic_json(ROOT / 'config.json', config)
    if (ROOT / 'rows.json').exists() and read(ROOT / 'rows.json') != rows:
        raise ValueError('Spinner training clips changed')
    atomic_json(ROOT / 'rows.json', rows)
    estimated_gib = sum(int(r['excerpt_length_ms'] / 8) * 13476 * 4 for r in rows) / 1024**3
    print(json.dumps({'event': 'v23_spinner_preflight', **manifest,
                      'estimated_cache_gib': estimated_gib,
                      'free_gib': shutil.disk_usage(ROOT).free / 1024**3}), flush=True)
    if args.preflight_only:
        return
    reserve_gib = 18 if args.purple_spinner else 23
    if shutil.disk_usage(ROOT).free < (estimated_gib + reserve_gib) * 1024**3:
        raise RuntimeError('Disk reserve too low for spinner fine-tune cache')
    engine = WarmValidationEngine(config, ROOT)
    cache = ROOT / 'cache'
    cache_parallel(engine, rows, cache, workers=2, stage='spinner')
    if missing_indices(rows, cache, config, 'clean'):
        raise RuntimeError('Incomplete spinner cache')
    if args.cache_only:
        return
    paths = entries(cache)
    # Keep previously learned circles in each epoch by replaying the existing
    # v23 caches alongside the new spinner-focused clips.
    old_a = entries(FULL / 'cache_A')
    old_c = entries(FULL / 'cache_C')
    rehearsal = old_a[::3] + old_c[::2]
    if args.purple_spinner:
        # A renderer switch changes features only when a swell is present.
        # Never mix historical blue-spinner features into the purple run.
        rows_a, rows_c = read(FULL / 'rows_A.json'), read(FULL / 'rows_C.json')
        safe = lambda path, rows: not any(note['type'] == 'swell'
                                          for note in rows[int(path.stem)]['notes'])
        rehearsal = [path for path in old_a[::3] if safe(path, rows_a)] + [
            path for path in old_c[::2] if safe(path, rows_c)]
    paths += rehearsal
    baseline, _ = load_best(INITIAL)
    baseline_reports = {'normal': engine.evaluate(normal, baseline),
                        'slow': engine.evaluate(slow, baseline),
                        'normal_spinner': engine.evaluate(normal_long, baseline),
                        'slow_spinner': engine.evaluate(slow_long, baseline)}
    atomic_json(ROOT / 'baseline.json', baseline_reports)
    width = baseline.mean.numel()
    policy = SensoryTemporalPolicy(width).cuda()
    policy.load_state_dict(baseline.state_dict())
    del baseline
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-5, weight_decay=.001)
    weights = torch.tensor(config['class_weights'], dtype=torch.float32, device='cuda')
    best = -1e9
    stale = 0
    start = 1
    checkpoint = ROOT / 'last.pt'
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location='cuda', weights_only=False)
        policy.load_state_dict(payload['policy'])
        optimizer.load_state_dict(payload['optimizer'])
        start, best, stale = payload['epoch'] + 1, payload['best_objective'], payload['stale']
    for epoch in range(start, args.epochs + 1):
        loss = train_epoch(policy, optimizer, paths, weights)
        policy.eval()
        reports = {'normal': engine.evaluate(normal, policy),
                   'slow': engine.evaluate(slow, policy),
                   'normal_spinner': engine.evaluate(normal_long, policy),
                   'slow_spinner': engine.evaluate(slow_long, policy)}
        spinner = .5 * (reports['normal_spinner']['swell']['coverage'] +
                        reports['slow_spinner']['swell']['coverage'])
        value = objective(reports['normal'], reports['slow']) + .3 * spinner
        guard = (reports['normal']['accuracy'] >= baseline_reports['normal']['accuracy'] - .005 and
                 reports['slow']['accuracy'] >= baseline_reports['slow']['accuracy'] - .03 and
                 reports['normal']['false_hits'] <= baseline_reports['normal']['false_hits'] + 10 and
                 reports['slow']['false_hits'] <= baseline_reports['slow']['false_hits'] + 10)
        improved = guard and value > best + 1e-5
        best, stale = (value, 0) if improved else (best, stale + 1)
        record = {'epoch': epoch, 'loss': loss, 'objective': value,
                  'guard_passed': guard, 'best': improved, **reports}
        atomic_json(ROOT / f'epoch-{epoch}.json', record)
        payload = {'architecture': config['architecture'], 'config': config,
                   'epoch': epoch, 'threshold': .65, 'feature_width': width,
                   'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                   'best_objective': best, 'stale': stale}
        torch.save(payload, checkpoint)
        if improved:
            torch.save(payload, ROOT / 'best.pt')
        print(json.dumps({'event': 'v23_spinner_epoch', 'epoch': epoch, 'loss': loss,
                          'normal_accuracy': reports['normal']['accuracy'],
                          'slow_accuracy': reports['slow']['accuracy'],
                          'normal_spinner_coverage': reports['normal_spinner']['swell']['coverage'],
                          'slow_spinner_coverage': reports['slow_spinner']['swell']['coverage'],
                          'guard': guard, 'best': improved}), flush=True)
        if epoch >= 4 and stale >= 4:
            print(json.dumps({'event': 'v23_spinner_early_stop', 'epoch': epoch}), flush=True)
            break
    if not (ROOT / 'best.pt').exists():
        atomic_json(ROOT / 'result.json', {'status': 'no_safe_improvement', 'baseline': baseline_reports})
        return
    selected, payload = load_best(ROOT / 'best.pt')
    full_normal = contextual_probes(split, prelude_ms=2000)
    full_slow = contextual_probes({'validation': slow_val}, prelude_ms=2000)
    full_reports = {'normal': engine.evaluate(full_normal, selected),
                    'slow': engine.evaluate(full_slow, selected)}
    original = read(FULL / 'result.json')['C_validation']
    safe = (full_reports['normal']['accuracy'] >= original['normal']['accuracy'] - .005 and
            full_reports['slow']['accuracy'] >= original['slow']['accuracy'] - .02 and
            full_reports['normal']['swell']['coverage'] > original['normal']['swell']['coverage'] and
            full_reports['slow']['swell']['coverage'] > original['slow']['swell']['coverage'])
    atomic_json(ROOT / 'result.json', {'status': 'candidate_only', 'passes_full_validation': safe,
                                      'epoch': payload['epoch'], 'checkpoint': str(ROOT / 'best.pt'),
                                      'baseline': baseline_reports,
                                      'full_validation': full_reports,
                                      'original_full_validation': original})
    print(json.dumps({'event': 'v23_spinner_complete', 'epoch': payload['epoch'],
                      'passes_full_validation': safe,
                      'normal_spinner_coverage': full_reports['normal']['swell']['coverage'],
                      'slow_spinner_coverage': full_reports['slow']['swell']['coverage']}), flush=True)


if __name__ == '__main__':
    main()
