"""Sandboxed-by-selection OSZ chart inference with selected validated models."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import traceback
from zipfile import ZipFile

import torch

from flytaiko.extract_beatmaps import extract_beatmap_data
from flytaiko.neural_campaign import atomic_json
from flytaiko.neural_system import seed_all
from taiko.parser import parse_osu_text
from flytaiko.replay_models import load_v25
from flytaiko.warm_validation import WarmValidationEngine
from flytaiko.replay_store import save_job


BASE = Path('runs/user_replays')
PUBLIC = Path('web-fly/public/demos')
KEYS = ['NONE', 'F', 'J', 'D', 'K', 'F+J', 'D+K']
STABLE_TAIKO_HR_16_9_SCROLL = 1.4 * ((16 / 9) / (4 / 3))


def apply_hard_rock(row, hp_drain=None):
    """Apply stable Taiko HR at the fixed 16:9 model/playfield reference.

    This does not claim complete osu!stable parity: HP drain, score, hitsounds,
    and client-specific visual effects are outside the FlyTaiko evaluator.
    """
    metadata = row['metadata']
    metadata['overall_difficulty'] = min(10.0, float(metadata['overall_difficulty']) * 1.4)
    if hp_drain is not None:
        metadata['hp_drain'] = min(10.0, float(hp_drain) * 1.4)
    metadata['mods'] = ['HR']
    row['ruleset_profile'] = {
        'reference': 'osu!stable taiko, 16:9',
        'scroll_multiplier': STABLE_TAIKO_HR_16_9_SCROLL,
        'judgment': 'piecewise Taiko OD windows in neural_game.taiko_windows',
        'swell': 'OD-dependent required hits',
        'not_simulated': ['HP drain/failure', 'score multiplier/scoring', 'client hitsounds'],
    }
    od = metadata['overall_difficulty']
    rate = (3 + .4 * od if od <= 5 else 5 + .5 * (od - 5)) * 1.65
    for note in row['notes']:
        note['scroll_px_per_ms'] *= STABLE_TAIKO_HR_16_9_SCROLL
        if note['type'] == 'swell':
            note['required_hits'] = max(1, int((note['end_t'] - note['t']) / 1000 * rate))
    return row


def update(folder, **fields):
    path = folder / 'status.json'
    old = json.loads(path.read_text())
    old.update(fields)
    atomic_json(path, old)
    save_job(old)


def audio_member(osu_text, chart_name, names):
    match = re.search(r'(?mi)^AudioFilename\s*:\s*(.+?)\s*$', osu_text)
    if not match:
        return None
    value = match.group(1).strip().replace('\\', '/')
    candidate = str(PurePosixPath(chart_name).parent / value)
    normalized = PurePosixPath(candidate)
    if normalized.is_absolute() or '..' in normalized.parts or normalized.suffix.lower() not in ('.mp3', '.ogg', '.wav'):
        return None
    found = [name for name in names if name.casefold() == candidate.casefold()]
    return found[0] if len(found) == 1 else None


def background_member(osu_text, chart_name, names):
    match = re.search(r'(?mi)^\s*0\s*,\s*0\s*,\s*"([^"\r\n]+)"', osu_text)
    if not match:
        return None
    candidate = str(PurePosixPath(chart_name).parent / match.group(1).replace('\\', '/'))
    path = PurePosixPath(candidate)
    if path.is_absolute() or '..' in path.parts or path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
        return None
    found = [name for name in names if name.casefold() == candidate.casefold()]
    return found[0] if len(found) == 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', required=True)
    parser.add_argument('--chart', type=int, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--mods', choices=['NM', 'HR', 'DT', 'DTHR'], default='NM')
    parser.add_argument('--model', choices=['v25'], default='v25')
    args = parser.parse_args()
    if not all(re.fullmatch(r'[a-f0-9]{12}', value) for value in (args.upload, args.job)):
        raise ValueError('Invalid job/upload identifier')
    seed_all(42)
    torch.set_num_threads(2)
    folder = BASE / args.job
    try:
        upload = json.loads((BASE / args.upload / 'upload.json').read_text())
        chart = upload['charts'][args.chart]
        archive = BASE / args.upload / 'uploaded.osz'
        if hashlib.sha256(archive.read_bytes()).hexdigest() != upload['sha256']:
            raise ValueError('Uploaded OSZ hash changed')
        with ZipFile(archive) as z:
            raw = z.read(chart['member'])
            text = raw.decode('utf-8-sig', errors='replace')
            parsed = parse_osu_text(text)
            if parsed is None or parsed.duration_ms > 600000:
                raise ValueError('Selected chart is not supported Taiko under 10 minutes')
            row = extract_beatmap_data(parsed)
            if 'HR' in args.mods:
                apply_hard_rock(row, parsed.metadata.hp_drain)
            if 'DT' in args.mods:
                row['ruleset_profile'] = {**row.get('ruleset_profile', {}),
                                          'clock_rate': 1.5,
                                          'clock_rate_note': 'chart clock 1.5x; neural frame/biophysical step remains 8ms'}
            row['metadata']['mods'] = [mod for mod in ('DT', 'HR') if mod in args.mods]
            row.update(source=f'uploaded.osz!{chart["member"]}', source_sha256=hashlib.sha256(raw).hexdigest())
            audio_name = audio_member(text, chart['member'], z.namelist())
        policy, payload, checkpoint, config = load_v25()
        config = {**config, 'clock_rate': 1.5 if 'DT' in args.mods else 1.0}
        update(folder, status='simulating', checkpoint=str(checkpoint), duration_ms=parsed.duration_ms,
               mods=args.mods, model=args.model, clock_rate=config['clock_rate'])
        engine = WarmValidationEngine(config, folder)
        metrics, games, trace = engine.run([row], policy, threshold=payload['threshold'], trace=True)
        replay = folder / 'replay.json'
        atomic_json(replay, {'schema_version': 4, 'metadata': row['metadata'],
                             'beatmap': row, 'metrics': metrics,
                             'actions': [{**action, 'action': KEYS[action['action_id']]}
                                         for action in games[0].actions],
                             'neural_trace': trace, 'provenance': config,
                             'checkpoint': str(checkpoint), 'mods': args.mods, 'model': args.model,
                             'clock_rate': config['clock_rate'],
                             'ruleset_profile': row.get('ruleset_profile')})
        update(folder, status='publishing', metrics=metrics)
        dataset = {'v23': 'user-v23-', 'v24-purple': 'user-v24-',
                   'v25': 'user-v25-'}[args.model] + args.job
        destination = PUBLIC / dataset
        command = [sys.executable, '-u', '-m', 'flytaiko.prepare_malecns_taiko_demo',
                   '--replay', str(replay), '--public', str(destination), '--no-audio']
        subprocess.run(command, check=True)
        if audio_name:
            with ZipFile(archive) as z:
                audio = z.read(audio_name)
            if len(audio) <= 64 * 1024**2:
                extension = PurePosixPath(audio_name).suffix.lower()
                audio_file = destination / ('audio' + extension)
                audio_file.write_bytes(audio)
                manifest_path = destination / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                manifest['audio_url'] = f'demos/{dataset}/{audio_file.name}'
                manifest['audio_sha256'] = hashlib.sha256(audio).hexdigest()
                manifest['audio_provenance'] = {'status': 'uploaded_osz',
                                                'chart_member': chart['member'],
                                                'audio_member': audio_name}
                atomic_json(manifest_path, manifest)
        with ZipFile(archive) as z:
            name = background_member(text, chart['member'], z.namelist())
            if name and z.getinfo(name).file_size <= 16 * 1024**2:
                image = z.read(name)
                target = destination / ('background' + PurePosixPath(name).suffix.lower())
                target.write_bytes(image)
                path = destination / 'manifest.json'
                manifest = json.loads(path.read_text())
                manifest['background_url'] = f'demos/{dataset}/{target.name}'
                atomic_json(path, manifest)
        if os.environ.get('FLYTAIKO_CLOUD_URL'):
            update(folder, status='publishing_cloud', metrics=metrics, dataset=dataset)
            subprocess.run(['node', 'web-fly/scripts/publish-replay.mjs', dataset, str(destination.resolve())], check=True)
        update(folder, status='complete', metrics=metrics, dataset=dataset,
               url=f'/malecns-taiko.html?dataset={dataset}')
        print(json.dumps({'event': 'user_replay_complete', 'job': args.job,
                          'dataset': dataset, 'metrics': metrics}), flush=True)
    except Exception as error:
        update(folder, status='failed', error=str(error)[:500])
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
