"""CPU-only presentation export: actual recorded keys, native judgments and trace.

Does not retrain, resimulate neurons, retime keys, or overwrite campaign outputs.
The existing evaluator reconstructs omitted auto-miss events from recorded keys;
all resulting gameplay metrics must equal the original replay before publishing.
"""
import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from flytaiko.prepare_malecns import ROOT
from flytaiko.visual_taiko import VisualGame, all_metrics, beatmap_from_json


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    temporary.replace(path)


def gameplay(value):
    if value['metrics']['profile'] != 'clean':
        raise ValueError('Only fixed 8ms clean recordings are supported by this export')
    game = VisualGame(beatmap_from_json(value['beatmap']))
    clock_rate = float(value.get('clock_rate', 1.))
    if not 0.5 <= clock_rate <= 2.0:
        raise ValueError('Invalid replay clock rate')
    game.current_time_ms = -8 * clock_rate
    source = value['actions']
    events = []
    cursor = 0
    event_cursor = 0
    deadline = max((n.time_ms for n in game.beatmap.notes), default=0) + 700000
    while not game.is_done:
        game.advance(8 * clock_rate)
        # Advance may expire circles before the key event in this same frame.
        for event in game.events[event_cursor:]:
            events.append({**event, 'time_ms': game.current_time_ms})
        event_cursor = len(game.events)
        while cursor < len(source) and source[cursor]['time_ms'] <= game.current_time_ms:
            action = source[cursor]
            if action['time_ms'] != game.current_time_ms:
                raise ValueError('Recorded action does not align with clean 8ms timeline')
            game.hit(action['action_id'])
            for event in game.events[event_cursor:]:
                events.append({**event, 'time_ms': game.current_time_ms})
            event_cursor = len(game.events)
            cursor += 1
        if game.current_time_ms > deadline:
            raise RuntimeError('Presentation timeline watchdog')
    if cursor != len(source):
        raise ValueError('Unconsumed replay actions')
    metrics = all_metrics([game])
    for name, expected in value['metrics'].items():
        if name in metrics and metrics[name] != expected:
            raise ValueError(f'Presentation evaluator differs: {name}: {metrics[name]} != {expected}')
    if len(game.actions) != len(source):
        raise ValueError('Key count changed during presentation export')
    for original, reconstructed in zip(source, game.actions):
        if any(original.get(k) != reconstructed.get(k) for k in
               ('time_ms', 'action_id', 'judgment', 'note_index', 'object_index')):
            raise ValueError('Recorded key judgment or target changed')
    circles = [n for n in value['beatmap']['notes'] if n['type'] in ('don', 'kat', 'don_big', 'kat_big')]
    notes = []
    circle_index = 0
    for index, note in enumerate(value['beatmap']['notes']):
        row = {**note, 'object_index': index}
        if note['type'] in ('don', 'kat', 'don_big', 'kat_big'):
            row['circle_index'] = circle_index
            row['judged_at_ms'] = events[circle_index]['time_ms']
            circle_index += 1
        notes.append(row)
    assert circle_index == len(circles) == len(events)
    source_export = (value['phase_a_export'] if 'phase_a_export' in value else
                     {'source': value['beatmap']['source'],
                      'source_sha256': value['beatmap']['source_sha256']})
    return {'schema_version': 1, 'metadata': value['metadata'], 'notes': notes,
            'actions': source, 'events': events, 'long_results': game.long_results,
            'duration_ms': game.current_time_ms, 'clock_rate': clock_rate,
            'mods': value.get('mods', 'NM'), 'metrics': value['metrics'],
            'checkpoint': value['checkpoint'],
            'source_export': source_export,
            'provenance': 'Recorded keys/timestamps/judgments; exact CPU evaluator parity; no hit snapping'}


def neural(value, destination, stride=4, url_prefix='demos/malecns-taiko'):
    if stride < 1:
        raise ValueError('Trace stride must be positive')
    frames = value['neural_trace'][::stride]
    if frames[-1]['time_ms'] != value['neural_trace'][-1]['time_ms']:
        frames.append(value['neural_trace'][-1])
    ids = frames[0]['body_ids']
    group_names = list(frames[0]['groups'])
    if not all(f['body_ids'] == ids and list(f['groups']) == group_names for f in frames):
        raise ValueError('Trace ID/group order changed')
    data = np.asarray([[f['rates_hz'], f['voltage_mv']] for f in frames], dtype='<f4')
    if data.shape != (len(frames), 2, len(ids)) or not np.isfinite(data).all():
        raise ValueError('Invalid recorded neural state')
    temporary = destination.with_suffix('.tmp')
    temporary.write_bytes(data.tobytes())
    temporary.replace(destination)
    return {'body_ids': ids, 'group_names': group_names,
            'time_ms': [f['time_ms'] for f in frames],
            'groups': [[[f['groups'][g]['rate_hz'], f['groups'][g]['voltage_mv'],
                         f['groups'][g]['spikes']] for g in group_names] for f in frames],
            'data_url': url_prefix + '/' + destination.name,
            'layout': 'little-endian float32 [frame, rate/voltage, sampled neuron]',
            'shape': list(data.shape), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'activity_units': frames[0]['activity_units'], 'neuron_model': frames[0]['neuron_model'],
            'provenance': f'Every {stride}th actual recorded frame plus final frame; no interpolated activity'}


def observations(value, trace, destination, url_prefix):
    """Pack the exact captured model RGB PNGs, without redrawing the beatmap."""
    camera = value.get('provenance', {})
    bounds = camera.get('retina_bounds', [0, 0, 1, 1])
    if len(bounds) != 4 or not (0 <= bounds[0] < bounds[2] <= 1 and
                                0 <= bounds[1] < bounds[3] <= 1):
        raise ValueError('Invalid recorded retina camera bounds')
    by_time = {frame['time_ms']: frame for frame in value['neural_trace']}
    packed = bytearray()
    entries = []
    dimensions = None
    for time_ms in trace['time_ms']:
        frame = by_time[time_ms]
        encoded = frame.get('rgb_png', '')
        if not encoded.startswith('data:image/png;base64,'):
            raise ValueError('Recorded model observation PNG is missing')
        png = base64.b64decode(encoded.split(',', 1)[1], validate=True)
        size = (int.from_bytes(png[16:20], 'big'), int.from_bytes(png[20:24], 'big'))
        if png[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' or \
                size not in ((512, 96), (1000, 300)) or png[24:26] != b'\x08\x02' or \
                (dimensions is not None and size != dimensions):
            raise ValueError('Unexpected recorded model RGB PNG dimensions or format')
        dimensions = size
        observation_time = frame['observation_time_ms']
        if observation_time > time_ms or time_ms - observation_time > 50:
            raise ValueError('Observation and neural trace times do not align')
        entries.append([len(packed), len(png), observation_time])
        packed.extend(png)
    temporary = destination.with_suffix('.tmp')
    temporary.write_bytes(packed)
    temporary.replace(destination)
    trace['observations'] = {
        'url': url_prefix + '/' + destination.name,
        'sha256': hashlib.sha256(packed).hexdigest(),
        'format': 'concatenated exact RGB PNGs', 'width': dimensions[0], 'height': dimensions[1],
        'retina_bounds': bounds, 'retina_bilinear': camera.get('retina_bilinear', False),
        'retina_x_warp': camera.get('retina_x_warp'),
        'frames': entries, 'provenance': 'Exact recorded model inputs at sampled neural times; no browser re-rendering',
    }


def audio_from_archive(value, archive):
    """Validate the recorded native chart, not an older OSZ chart revision.

    The user upload is .osu-only. The existing OSZ supplies this song's audio;
    its older chart is never used for note timing, SV, or judgments.
    """
    from taiko.parser import parse_osu_text
    from flytaiko.extract_beatmaps import extract_beatmap_data
    recorded = value.get('phase_a_export', value['beatmap'])
    native_bytes = Path(recorded['source']).read_bytes()
    if hashlib.sha256(native_bytes).hexdigest() != recorded['source_sha256']:
        raise ValueError('Recorded native source hash mismatch')
    native_text = native_bytes.decode('utf-8-sig')
    native = parse_osu_text(native_text)
    if native is None or extract_beatmap_data(native)['notes'] != value['beatmap']['notes']:
        raise ValueError('Recorded native chart differs from replay')
    chart = next(n for n in archive.namelist() if n.endswith('[CATASTROPHE].osu'))
    archived_bytes = archive.read(chart)
    archived_text = archived_bytes.decode('utf-8-sig')
    archived = parse_osu_text(archived_text)
    if archived is None or any(getattr(native.metadata, k) != getattr(archived.metadata, k)
                               for k in ('title', 'artist', 'version')):
        raise ValueError('Audio archive belongs to a different song/difficulty')

    def field(text, name, default=None):
        match = re.search(r'^' + re.escape(name) + r'\s*:\s*(.*?)\s*$', text, re.MULTILINE)
        return match.group(1) if match else default

    audio_name = field(native_text, 'AudioFilename')
    if not audio_name or field(archived_text, 'AudioFilename') != audio_name:
        raise ValueError('Audio filename does not match the recorded song')
    if field(native_text, 'AudioLeadIn', '0') != field(archived_text, 'AudioLeadIn', '0'):
        raise ValueError('Audio lead-in differs between song revisions')
    return archive.read(audio_name), {
        'archive': 'demos/ideoless.osz', 'filename': audio_name,
        'chart_revision_matches': hashlib.sha256(archived_bytes).hexdigest() == recorded['source_sha256'],
        'recorded_native_source_verified': True,
        'note_timing_source': 'Recorded uploaded native chart; OSZ chart is NOT used',
        'audio_identity': 'Same title/artist/difficulty/audio filename/lead-in; original PC audio bytes not uploaded',
    }


def measured_sample_edges(body_ids, destination, maximum_per_source=5,
                          url_prefix='demos/malecns-taiko'):
    """Actual outgoing measured contacts from traced IDs to located somata.

    Only the source neuron has a recorded activity trace. A target position
    means anatomical geometry, NEVER inferred target activity.
    """
    from scipy import sparse
    arrays = np.load(ROOT / 'graph_arrays.npz', allow_pickle=False)
    lookup = {str(body_id): i for i, body_id in enumerate(arrays['ids'])}
    neurons = json.loads((ROOT / 'neurons.json').read_text())
    contacts = sparse.load_npz(ROOT / 'contacts_pre_post.npz').tocsr()
    edges = []
    for sample_index, body_id in enumerate(body_ids):
        if body_id not in lookup:
            raise ValueError('Sample ID absent from measured graph')
        source = lookup[body_id]
        start, end = contacts.indptr[source:source+2]
        choices = sorted(((int(contacts.indices[j]), float(contacts.data[j]))
                          for j in range(start, end) if neurons[int(contacts.indices[j])]['soma'] is not None),
                         key=lambda pair: (-pair[1], pair[0]))[:maximum_per_source]
        for target, weight in choices:
            if weight <= 0 or neurons[source]['soma'] is None:
                continue
            edges.append([sample_index, neurons[target]['soma'], weight])
    write_json(destination, edges)
    return {'url': url_prefix + '/' + destination.name,
            'count': len(edges), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'provenance': 'Top measured outgoing contacts from each traced source to located soma; target activity is NOT recorded'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', type=Path, default=Path('web-fly/public/demos/malecns-taiko'))
    parser.add_argument('--source-dir', type=Path, default=Path('web-fly/public/demos'))
    parser.add_argument('--trace-stride', type=int, default=1,
                        help='Keep all recorded neural frames by default; display FPS is independent')
    parser.add_argument('--replay', type=Path,
                        help='Export a completed full replay into a separate presentation directory')
    parser.add_argument('--no-audio', action='store_true',
                        help='Publish a chart-only replay when the native song audio was not uploaded')
    parser.add_argument('--observations-only', action='store_true',
                        help='Add captured model RGB frames to an already exported full trace')
    args = parser.parse_args()
    args.public.mkdir(parents=True, exist_ok=True)
    if args.replay:
        prefix = args.public.resolve().relative_to(Path('web-fly/public').resolve()).as_posix()
        if not re.fullmatch(r'demos/[A-Za-z0-9_.-]+', prefix):
            raise ValueError('Replay output must be a single safe demos directory')
        value = json.loads(args.replay.read_text())
        if args.observations_only:
            trace_path = args.public / 'full-trace.json'
            trace = json.loads(trace_path.read_text())
            observations(value, trace, args.public / 'full-observations.bin', prefix)
            write_json(trace_path, trace)
            print(json.dumps({'exported': 'observations', 'frames': len(trace['observations']['frames'])}), flush=True)
            return
        game = gameplay(value)
        write_json(args.public / 'full-gameplay.json', game)
        trace = neural(value, args.public / 'full-neurons.bin', args.trace_stride, prefix)
        observations(value, trace, args.public / 'full-observations.bin', prefix)
        neurons = json.loads((ROOT / 'neurons.json').read_text())
        by_id = {n['body_id']: n for n in neurons}
        if not all(i in by_id for i in trace['body_ids']):
            raise ValueError('Unmeasured neuron ID')
        trace['sample_soma'] = [by_id[i]['soma'] for i in trace['body_ids']]
        write_json(args.public / 'full-trace.json', trace)
        positions = np.asarray([n['soma'] for n in neurons if n['soma'] is not None], dtype='<i4')
        (args.public / 'anatomy.bin').write_bytes(positions.tobytes())
        if args.no_audio:
            audio_fields = {'audio_url': None, 'audio_sha256': None,
                            'audio_provenance': {'status': 'not_uploaded',
                                                 'reason': 'Native upload contained only the .osu chart'}}
        else:
            with ZipFile(args.source_dir / 'ideoless.osz') as archive:
                audio, audio_provenance = audio_from_archive(value, archive)
            (args.public / 'audio.mp3').write_bytes(audio)
            audio_fields = {'audio_url': prefix + '/audio.mp3',
                            'audio_sha256': hashlib.sha256(audio).hexdigest(),
                            'audio_provenance': audio_provenance}
        edges = measured_sample_edges(trace['body_ids'], args.public / 'sample-edges.json',
                                      url_prefix=prefix)
        version = re.search(r'(?:malecns|playing-god)-(v\d+)-', args.replay.name)
        replay_version = version.group(1) if version else 'selected'
        write_json(args.public / 'manifest.json', {
            'scope': f'Full native-map replay from the selected {replay_version} checkpoint',
            'mods': value.get('mods', 'NM'),
            'clock_rate': value.get('clock_rate', 1.0),
            'ruleset_profile': value.get('ruleset_profile'),
            'replays': [{'kind': 'full', 'label': f"{value['metadata']['title']} [{value.get('mods', 'NM')}] · {replay_version} · full map",
                         'gameplay_url': prefix + '/full-gameplay.json',
                         'trace_url': prefix + '/full-trace.json', 'metrics': game['metrics'],
                         'original_replay_sha256': hashlib.sha256(args.replay.read_bytes()).hexdigest()}],
            **audio_fields, 'sample_edges': edges,
            'anatomy': {'url': prefix + '/anatomy.bin', 'count': len(positions),
                        'layout': 'little-endian int32 [located soma, xyz]', 'units': '8 nm voxel',
                        'sha256': hashlib.sha256(positions.tobytes()).hexdigest()},
            'disclaimer': 'Measured topology/IDs; experimental graded-rate dynamics, not biological Hz/mV/spikes'})
        print(json.dumps({'exported': 'full', 'metrics': game['metrics'],
                          'trace_shape': trace['shape'], 'manifest': str(args.public / 'manifest.json')}), flush=True)
        return
    source_manifest = json.loads((args.source_dir / 'malecns-phase-a-replays.json').read_text())
    neurons = json.loads((ROOT / 'neurons.json').read_text())
    by_id = {n['body_id']: n for n in neurons}
    positions = np.asarray([n['soma'] for n in neurons if n['soma'] is not None], dtype='<i4')
    (args.public / 'anatomy.bin').write_bytes(positions.tobytes())
    entries = []
    # Preview first: opening the page must not force a 155MB JSON download.
    for kind in ('preview', 'full'):
        source = args.source_dir / f'malecns-phase-a-ideology-{kind}.json'
        value = json.loads(source.read_text())
        game = gameplay(value)
        write_json(args.public / f'{kind}-gameplay.json', game)
        trace = neural(value, args.public / f'{kind}-neurons.bin', args.trace_stride)
        observations(value, trace, args.public / f'{kind}-observations.bin', 'demos/malecns-taiko')
        if not all(i in by_id for i in trace['body_ids']):
            raise ValueError('Unmeasured neuron ID')
        trace['sample_soma'] = [by_id[i]['soma'] for i in trace['body_ids']]
        write_json(args.public / f'{kind}-trace.json', trace)
        entries.append({'kind': kind, 'label': 'IDEALESS IDEOLOGY · Phase A epoch 12 · ' +
                        ('35 giây đầu' if kind == 'preview' else 'toàn bài'),
                        'gameplay_url': f'demos/malecns-taiko/{kind}-gameplay.json',
                        'trace_url': f'demos/malecns-taiko/{kind}-trace.json',
                        'metrics': game['metrics'],
                        'original_replay_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        print(json.dumps({'exported': kind, 'metrics': game['metrics'],
                          'trace_shape': trace['shape']}), flush=True)
    with ZipFile(args.source_dir / 'ideoless.osz') as archive:
        audio, audio_provenance = audio_from_archive(value, archive)
        (args.public / 'audio.mp3').write_bytes(audio)
    edge_manifest = measured_sample_edges(json.loads((args.public / 'preview-trace.json').read_text())['body_ids'],
                                          args.public / 'sample-edges.json')
    write_json(args.public / 'manifest.json', {
        'scope': source_manifest['scope'], 'checkpoint_epoch': 12, 'replays': entries,
        'audio_url': 'demos/malecns-taiko/audio.mp3',
        'audio_sha256': hashlib.sha256(audio).hexdigest(),
        'audio_provenance': audio_provenance,
        'sample_edges': edge_manifest,
        'anatomy': {'url': 'demos/malecns-taiko/anatomy.bin', 'count': len(positions),
                    'layout': 'little-endian int32 [located soma, xyz]', 'units': '8 nm voxel',
                    'sha256': hashlib.sha256(positions.tobytes()).hexdigest()},
        'disclaimer': 'Measured topology/IDs; experimental graded-rate dynamics, not biological Hz/mV/spikes'})


if __name__ == '__main__':
    main()
