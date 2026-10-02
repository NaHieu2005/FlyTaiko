"""Managed A -> gated B campaign with song-disjoint data and metric logs."""
import argparse
import json
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from flytaiko.flytaiko_pipeline import (Config, JsonBeatmap, Runtime, checkpoint, evaluate,
                              iter_frames, load_model, seed_all, train)
from taiko.environment import TaikoEnvironment

_worker_runtime = None


def init_worker(config):
    global _worker_runtime
    torch.set_num_threads(1)
    seed_all(config.seed)
    _worker_runtime = Runtime(config, 'cuda')


def collect_worker(task):
    index, raw, destination = task
    runtime = _worker_runtime
    states, labels = [], []
    for _, _, vector, expert in iter_frames(runtime, JsonBeatmap(raw)):
        states.append(vector.cpu())
        labels.append(expert)
        if len(states) % 2000 == 0:
            print(json.dumps({'event': 'parallel_progress', 'map': index + 1,
                              'frames': len(states)}), flush=True)
    path = Path(destination)
    temporary = path.with_suffix('.partial')
    torch.save({'states': torch.stack(states), 'labels': labels}, temporary)
    temporary.replace(path)
    return {'event': 'features_complete', 'map': index + 1, 'frames': len(states),
            'expert_hits': sum(x > 0 for x in labels)}


def prepare_parallel(rows, directory, config, workers):
    directory.mkdir(parents=True, exist_ok=True)
    tasks = [(i, row, str(directory / f'{i:04d}.pt')) for i, row in enumerate(rows)
             if not (directory / f'{i:04d}.pt').exists()]
    print(json.dumps({'event': 'parallel_start', 'workers': workers,
                      'cached_maps': len(rows) - len(tasks), 'pending_maps': len(tasks)}), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=init_worker, initargs=(config,)) as pool:
        futures = [pool.submit(collect_worker, task) for task in tasks]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


def emit(path, row):
    print(json.dumps(row), flush=True)
    with path.open('a') as handle:
        handle.write(json.dumps(row) + '\n')


def select_short_campaign(raw, previous=None, limit_ms=600000):
    """Preserve existing slots while replacing songs without an eligible map."""
    eligible = {}
    for row in raw:
        notes = row.get('notes', [])
        if len(notes) < 100 or max(n['t'] for n in notes) >= limit_ms:
            continue
        meta = row.get('metadata', {})
        key = (meta.get('artist'), meta.get('title'))
        if key not in eligible or len(notes) > len(eligible[key]['notes']):
            eligible[key] = row
    candidates = list(eligible.values())
    random.Random(42).shuffle(candidates)
    if not previous:
        if len(candidates) < 170:
            raise ValueError('Need 170 distinct songs below duration limit')
        return candidates[:100], candidates[100:150], candidates[150:170]
    used = {(m.get('artist'), m.get('title')) for group in ('phase_a', 'phase_b', 'validation')
            for m in previous[group]}
    replacements = iter(r for r in candidates if
                        (r['metadata'].get('artist'), r['metadata'].get('title')) not in used)
    groups = []
    for group in ('phase_a', 'phase_b', 'validation'):
        selected = []
        for old in previous[group]:
            key = (old.get('artist'), old.get('title'))
            # Preserve the exact original difficulty when it meets the limit.
            match = next((r for r in raw if r.get('metadata') == old and r.get('notes')
                          and max(n['t'] for n in r['notes']) < limit_ms), None)
            selected.append(match or eligible.get(key) or next(replacements))
        groups.append(selected)
    return tuple(groups)


def phase_b(runtime, maps, validation, root, baseline):
    """Frozen reservoir, readout-only DAgger mixed equally with A examples."""
    out = root / 'phase_b'
    out.mkdir(exist_ok=True)
    optimizer = torch.optim.Adam(runtime.decoder.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    baseline_score = baseline['great_rate'] - baseline['miss_rate']
    best = baseline_score
    baseline_state = {k: v.detach().cpu().clone() for k, v in runtime.decoder.state_dict().items()}
    anchors = sorted((root / 'phase_a' / 'features').glob('*.pt'))
    for index, beatmap in enumerate(maps):
        runtime.reset(); runtime.decoder.eval()
        env = TaikoEnvironment(beatmap)
        states, labels = [], []
        last_hit = -9999
        while not env.is_done:
            state = env.get_state()
            expert = env.get_expert_action()
            vector = runtime.state_vector(state)
            with torch.no_grad():
                action = int(runtime.decoder(vector).argmax())
            if action and env.current_time_ms - last_hit < runtime.cfg.cooldown_ms:
                action = 0
            if action:
                last_hit = env.current_time_ms
            states.append(vector.cpu()); labels.append(expert)
            env.step(action)
        dagger = torch.stack(states)
        anchor = torch.load(anchors[index % len(anchors)], weights_only=True)
        n = min(len(dagger), len(anchor['states']), 8192)
        di = torch.randperm(len(dagger))[:n]
        ai = torch.randperm(len(anchor['states']))[:n]
        x = torch.cat((dagger[di], anchor['states'][ai])).to(runtime.device)
        y = torch.cat((torch.tensor(labels)[di], torch.tensor(anchor['labels'])[ai])).to(runtime.device)
        runtime.decoder.train()
        losses = []
        for ids in torch.randperm(len(y), device=runtime.device).split(64):
            loss = criterion(runtime.decoder(x[ids]), y[ids])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite B loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(runtime, validation)
        score = metrics['great_rate'] - metrics['miss_rate']
        accepted = score > best and metrics['miss_rate'] <= baseline['miss_rate']
        if accepted:
            best = score
            saved = checkpoint(runtime, runtime.cfg, index + 1, optimizer, best)
            saved['phase'] = 'B'
            torch.save(saved, out / 'best.pt')
            baseline_state = {k: v.detach().cpu().clone() for k, v in runtime.decoder.state_dict().items()}
        else:
            runtime.decoder.load_state_dict(baseline_state)
            optimizer = torch.optim.Adam(runtime.decoder.parameters(), lr=1e-5)
        emit(out / 'train.jsonl', {'episode': index + 1, 'loss': float(np.mean(losses)),
                                 'validation': metrics, 'accepted': accepted})
        if score < baseline_score - .15:
            emit(out / 'train.jsonl', {'event': 'stop', 'reason': 'B regression exceeds 15 percentage points'})
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/campaign_100A_50B'))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-map-minutes', type=float, default=10)
    args = parser.parse_args()
    root = args.root; root.mkdir(parents=True, exist_ok=True)
    seed_all(42)
    raw = json.load(open('kaggle_data/taiko_beatmaps.json'))
    split_path = root / 'split.json'
    previous = json.loads(split_path.read_text()) if split_path.exists() else None
    a, b, val = select_short_campaign(raw, previous, args.max_map_minutes * 60000)
    if previous:
        changes = [(group, i, old, row['metadata'])
                   for group, rows in [('phase_a', a), ('phase_b', b), ('validation', val)]
                   for i, (old, row) in enumerate(zip(previous[group], rows))
                   if old != row['metadata']]
        if changes:
            archive = root / 'history' / datetime.now().strftime('%Y%m%d_%H%M%S')
            archive.mkdir(parents=True)
            split_path.rename(archive / 'split.json')
            for group, index, old, new in changes:
                if group == 'phase_a':
                    cache = root / 'phase_a' / 'features' / f'{index:04d}.pt'
                    if cache.exists():
                        cache.rename(archive / cache.name)
                print(json.dumps({'event': 'map_replaced', 'group': group, 'map': index + 1,
                                  'old': old, 'new': new}), flush=True)
    manifest = {'seed': 42, 'phase_a': [r['metadata'] for r in a],
                'phase_b': [r['metadata'] for r in b], 'validation': [r['metadata'] for r in val],
                'max_map_minutes': args.max_map_minutes}
    (root / 'split.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'event': 'start', 'phase_a_songs':len(a), 'phase_b_songs':len(b),
                      'validation_songs':len(val)}), flush=True)
    cfg = Config()
    a_maps = [JsonBeatmap(r) for r in a]; validation = [JsonBeatmap(r) for r in val]
    a_dir = root / 'phase_a'; a_dir.mkdir(exist_ok=True)
    prepare_parallel(a, a_dir / 'features', cfg, args.workers)
    torch.set_num_threads(2)
    seed_all(cfg.seed)
    runtime = Runtime(cfg, 'cuda')
    last = a_dir / 'last.pt'
    train(runtime, a_maps, validation, args.epochs, 1e-3, a_dir / 'best.pt',
          last if last.exists() else None, None)
    load_model(runtime, a_dir / 'best.pt')
    saved = torch.load(a_dir / 'best.pt', map_location='cpu', weights_only=False)
    history = [json.loads(line) for line in (a_dir / 'train.jsonl').read_text().splitlines()]
    matching = [row['validation'] for row in history if row['epoch'] == saved['epoch']]
    metrics = matching[-1] if matching else evaluate(runtime, validation)
    (a_dir / 'baseline.json').write_text(json.dumps(metrics, indent=2))
    if metrics['great_rate'] + metrics['good_rate'] < .8 or metrics['great_rate'] < .7:
        emit(root / 'status.jsonl', {'event': 'stopped_before_B', 'reason': 'A validation gate failed',
                                  'validation': metrics})
        return
    phase_b(runtime, [JsonBeatmap(r) for r in b], validation, root, metrics)
    emit(root / 'status.jsonl', {'event': 'completed'})


if __name__ == '__main__':
    main()
