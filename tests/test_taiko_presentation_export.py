import numpy as np
import pytest
import hashlib
from zipfile import ZipFile
from prepare_malecns_taiko_demo import gameplay, neural, audio_from_archive
from visual_taiko import VisualGame, beatmap_from_json, all_metrics


def recorded_value():
    row = {'metadata': {'title': 'fixture', 'overall_difficulty': 5}, 'notes': [
        {'t': 100, 'type': 'don', 'end_t': 0},
        {'t': 300, 'type': 'kat', 'end_t': 0},
        {'t': 500, 'type': 'don_big', 'end_t': 0},
        {'t': 700, 'type': 'swell', 'end_t': 900, 'required_hits': 2},
        {'t': 1000, 'type': 'drumroll', 'end_t': 1200, 'tick_spacing_ms': 100},
    ]}
    game = VisualGame(beatmap_from_json(row))
    actions = {104: 1, 304: 1, 704: 1, 720: 3, 1000: 2, 1104: 1, 1200: 2}
    while not game.is_done:
        game.advance(8)
        if game.current_time_ms in actions:
            game.hit(actions[game.current_time_ms])
    return {'beatmap': row, 'metadata': row['metadata'], 'actions': game.actions,
            'metrics': {**all_metrics([game]), 'profile': 'clean'},
            'checkpoint': 'actual.pt', 'phase_a_export': {'checkpoint_epoch': 12}}


def test_presentation_preserves_wrong_color_auto_miss_and_native_longs():
    value = recorded_value()
    result = gameplay(value)
    assert result['actions'] == value['actions']
    assert [e['judgment'] for e in result['events']] == ['great', 'miss', 'miss']
    assert [e['reason'] for e in result['events']] == ['hit', 'wrong_color', 'unhit']
    assert result['events'][0]['time_ms'] == 104
    assert result['events'][0]['timing_error_ms'] == 4
    assert result['notes'][1]['judged_at_ms'] == 304
    assert result['metrics']['swell']['completed'] == 1
    assert result['metrics']['drumroll']['completed'] == 1


def test_double_time_replay_uses_12ms_chart_steps_for_8ms_real_frames():
    row = {'metadata': {'title': 'DT fixture', 'overall_difficulty': 10},
           'notes': [{'t': 120, 'type': 'don'}, {'t': 360, 'type': 'kat'}]}
    game = VisualGame(beatmap_from_json(row))
    game.current_time_ms = -12
    while not game.is_done:
        game.step_ms = 12
        game.advance(12)
        if game.current_time_ms == 120:
            game.hit(1)
        elif game.current_time_ms == 360:
            game.hit(3)
    value = {'beatmap': row, 'metadata': row['metadata'], 'actions': game.actions,
             'metrics': {**all_metrics([game]), 'profile': 'clean'},
             'clock_rate': 1.5, 'mods': 'DT', 'checkpoint': 'fixture.pt',
             'phase_a_export': {'checkpoint_epoch': 1}}
    result = gameplay(value)
    assert result['clock_rate'] == 1.5 and result['mods'] == 'DT'
    assert [event['judgment'] for event in result['events']] == ['great', 'great']
    assert [action['time_ms'] for action in result['actions']] == [120, 360]


def test_completed_replay_without_phase_a_export_keeps_source_provenance(monkeypatch):
    value = recorded_value()
    del value['phase_a_export']
    from prepare_malecns_taiko_demo import beatmap_from_json as parse
    monkeypatch.setattr('prepare_malecns_taiko_demo.beatmap_from_json',
                        lambda row: parse({k: v for k, v in row.items()
                                           if k not in ('source', 'source_sha256')}))
    value['beatmap']['source'] = '/verified/native.osu'
    value['beatmap']['source_sha256'] = 'a' * 64
    result = gameplay(value)
    assert result['source_export'] == {'source': '/verified/native.osu',
                                       'source_sha256': 'a' * 64}


def test_presentation_refuses_different_metrics():
    value = recorded_value()
    value['metrics']['miss'] += 1
    with pytest.raises(ValueError, match='Presentation evaluator differs'):
        gameplay(value)


def test_trace_binary_copies_real_samples_without_interpolation(tmp_path):
    frames = [{'time_ms': i*48, 'body_ids': ['actual-1', 'actual-2'],
               'rates_hz': [float(i), float(i+1)], 'voltage_mv': [-52., -49.],
               'groups': {'motor': {'rate_hz': float(i), 'voltage_mv': -50., 'spikes': 0}},
               'activity_units': 'proxy', 'neuron_model': 'hypothesis'} for i in range(10)]
    path = tmp_path / 'trace.bin'
    meta = neural({'neural_trace': frames}, path)
    assert meta['time_ms'] == [0, 192, 384, 432]
    data = np.frombuffer(path.read_bytes(), dtype='<f4').reshape(meta['shape'])
    assert data[:, 0, 0].tolist() == [0., 4., 8., 9.]
    assert data[:, 1, 1].tolist() == [-49.]*4


def audio_fixture(tmp_path, archived_filename='audio.mp3'):
    from taiko.parser import parse_osu_text
    from extract_beatmaps import extract_beatmap_data
    text = '''osu file format v14
[General]
AudioFilename: audio.mp3
AudioLeadIn: 0
Mode: 1
[Metadata]
Title: fixture
Artist: test
Version: CATASTROPHE
[Difficulty]
OverallDifficulty: 5
SliderMultiplier: 1.4
[TimingPoints]
0,500,4,2,1,50,1,0
[HitObjects]
256,192,1000,1,0,0:0:0:0:
'''
    source = tmp_path / 'native.osu'
    source.write_text(text)
    value = {'beatmap': extract_beatmap_data(parse_osu_text(text)),
             'phase_a_export': {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}
    archive_path = tmp_path / 'audio.osz'
    with ZipFile(archive_path, 'w') as archive:
        # Old chart revision must NOT supply native note times or SV.
        old = text.replace('256,192,1000', '256,192,1008').replace('audio.mp3', archived_filename)
        archive.writestr('fixture [CATASTROPHE].osu', old)
        archive.writestr(archived_filename, b'fixture-audio')
    return value, archive_path


def test_audio_archive_older_chart_is_not_used_for_recorded_timing(tmp_path):
    value, path = audio_fixture(tmp_path)
    with ZipFile(path) as archive:
        data, provenance = audio_from_archive(value, archive)
    assert data == b'fixture-audio'
    assert provenance['recorded_native_source_verified']
    assert not provenance['chart_revision_matches']
    assert value['beatmap']['notes'][0]['t'] == 1000


def test_audio_export_refuses_changed_native_source(tmp_path):
    value, path = audio_fixture(tmp_path)
    value['phase_a_export']['source_sha256'] = '0' * 64
    with ZipFile(path) as archive, pytest.raises(ValueError, match='source hash mismatch'):
        audio_from_archive(value, archive)


def test_audio_export_refuses_different_audio_filename(tmp_path):
    value, path = audio_fixture(tmp_path, 'different.mp3')
    with ZipFile(path) as archive, pytest.raises(ValueError, match='filename does not match'):
        audio_from_archive(value, archive)
