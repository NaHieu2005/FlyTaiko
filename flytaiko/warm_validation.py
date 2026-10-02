"""Context-preserving validation for v18 without changing cache/source physics.

Each original 8-second probe is simulated with 2 seconds of earlier gameplay.
Only the original probe interval contributes to its reported metrics.
"""
import copy
from types import SimpleNamespace

import numpy as np

from flytaiko.malecns_training import Engine, probes
from flytaiko.visual_taiko import all_metrics


CIRCLE_TYPES = {'don', 'kat', 'don_big', 'kat_big'}


def contextual_probes(split, prelude_ms=2000):
    originals = {row['source']: row for row in split['validation']}
    result = []
    for old in probes(split['validation'], length=8000):
        source = originals[old['source']]
        old_start = old['excerpt_start_ms']
        new_start = max(0, old_start - prelude_ms)
        end = old_start + old['excerpt_length_ms']
        notes = [copy.deepcopy(note) for note in source['notes']
                 if new_start <= note['t'] < end and
                 (not note.get('end_t') or note['end_t'] < end)]
        for note in notes:
            note['t'] -= new_start
            if note.get('end_t'):
                note['end_t'] -= new_start
        shifted = {**source, 'notes': notes, 'excerpt_start_ms': new_start,
                   'excerpt_length_ms': end - new_start,
                   'score_start_ms': old_start - new_start,
                   'score_end_ms': end - new_start}
        scored = sorted((round(note['t'] - shifted['score_start_ms'], 5), note['type'])
                        for note in shifted['notes']
                        if shifted['score_start_ms'] <= note['t'] < shifted['score_end_ms'])
        expected = sorted((round(note['t'], 5), note['type']) for note in old['notes'])
        if scored != expected:
            raise ValueError('Contextual validation changed scored notes')
        result.append(shifted)
    return result


def scored_view(game, row):
    start, end = row['score_start_ms'], row['score_end_ms']
    events = [event for event in game.events
              if start <= event['note_time_ms'] < end]
    false_hits = sum(action.get('judgment') == 'ignore' and
                     action.get('object_index') is None and
                     start <= action['time_ms'] < end for action in game.actions)
    long_indices = {obj['index'] for obj in game.long
                    if start <= obj['note'].time_ms < end and
                    obj['note'].end_time_ms < end}
    long_results = [record for record in game.long_results
                    if record['object_index'] in long_indices]
    return SimpleNamespace(events=events, false_hits=false_hits,
                           alt_violations=game.alt_violations,
                           long_results=long_results)


class WarmValidationEngine(Engine):
    def evaluate(self, rows, policy, profile='clean', threshold=.65, disconnect=False):
        if not rows or not all('score_start_ms' in row for row in rows):
            return super().evaluate(rows, policy, profile, threshold, disconnect)
        reports, views = [], []
        for offset in range(0, len(rows), self.config['batch']):
            selected = rows[offset:offset + self.config['batch']]
            report, games, _ = self.run(selected, policy, profile, threshold,
                                        disconnect, seed=42 + offset)
            reports.append(report)
            views.extend(scored_view(game, row) for game, row in zip(games, selected))
        result = all_metrics(views)
        result.update(profile=profile,
                      inference_batch_p95_ms=max(r['inference_batch_p95_ms'] for r in reports),
                      inference_batch_p50_ms=float(np.median(
                          [r['inference_batch_p50_ms'] for r in reports])))
        return result
