"""Timestamped Taiko judgments with explicit accounting for every note."""
from collections import deque
import copy
import math
import numpy as np
from taiko.environment import ACTION_NAMES, DON_ACTIONS, KAT_ACTIONS


def taiko_windows(od):
    """ppy/osu TaikoHitWindows: piecewise OD ranges, floor minus 0.5 ms."""
    od=float(np.clip(od,0,10))
    return {'great':math.floor(50-3*od)-.5,
            'good':math.floor(120-8*od if od<=5 else 110-6*od)-.5,
            'miss':math.floor(135-8*od if od<=5 else 120-5*od)-.5}


class Game:
    def __init__(self, beatmap, step_ms=8):
        self.beatmap = beatmap
        self.hit_notes = [n for n in beatmap.notes if n.note_type in ('don','kat','don_big','kat_big')]
        self.windows = taiko_windows(beatmap.metadata.overall_difficulty)
        self.step_ms = step_ms
        # One pre-roll observation interval makes a note at t=0 trainable,
        # while the first neural decision is still timestamped at t=0.
        self.current_time_ms = -float(step_ms)
        self.next_note_idx = 0
        self.don_alt = self.kat_alt = False
        self.combo = self.max_combo = 0
        self.events = []
        self.actions = []
        self.false_hits = self.alt_violations = 0
        self.last_hand = {'don':None,'kat':None}

    @property
    def is_done(self):
        return self.next_note_idx >= len(self.hit_notes)

    def expire(self):
        while not self.is_done and self.current_time_ms > self.hit_notes[self.next_note_idx].time_ms + self.windows['miss']:
            self.record('miss', 0, None, 'unhit')

    def record(self, judgment, action, error, reason):
        index = self.next_note_idx; note = self.hit_notes[index]
        self.events.append({'note_index':index, 'note_type':note.note_type,
                            'note_time_ms':note.time_ms, 'judgment':judgment,
                            'action':action, 'timing_error_ms':error, 'reason':reason})
        self.next_note_idx += 1
        self.combo = self.combo + 1 if judgment != 'miss' else 0
        self.max_combo = max(self.max_combo, self.combo)

    def hit(self, action):
        if not action:
            return
        if action not in range(1,7):
            raise ValueError('Invalid Taiko action')
        self.expire()
        color = 'don' if action in DON_ACTIONS else 'kat'
        if action in (1,2,3,4):
            hand = action in (2,4)
            self.alt_violations += int(self.last_hand[color] == hand)
            self.last_hand[color] = hand
            if color == 'don': self.don_alt = not hand
            else: self.kat_alt = not hand
        entry = {'time_ms':self.current_time_ms,'model_time_ms':self.current_time_ms,
                 'action':ACTION_NAMES[action], 'action_id':action, 'judgment':'ignore'}
        self.actions.append(entry)
        if self.is_done:
            self.false_hits += 1; return
        note = self.hit_notes[self.next_note_idx]
        error = self.current_time_ms - note.time_ms
        if abs(error) > self.windows['miss']:
            self.false_hits += 1; return
        same_color = (action in DON_ACTIONS) == note.is_don
        correct_size = (action in (5,6)) == note.is_big
        if not same_color:
            judgment, reason = 'miss', 'wrong_color'
        elif abs(error) <= self.windows['great']:
            judgment, reason = 'great', 'hit'
        elif abs(error) <= self.windows['good']:
            judgment, reason = 'good', 'hit'
        else:
            judgment, reason = 'miss', 'late_or_early'
        entry['judgment'] = judgment
        entry['note_index'] = self.next_note_idx
        self.record(judgment, action, error, reason)
        self.events[-1]['size_correct']=correct_size

    def expert(self, anticipation_ms=0):
        if self.is_done:
            return 0
        note = self.hit_notes[self.next_note_idx]
        dt = note.time_ms - self.current_time_ms - anticipation_ms
        if not -self.step_ms/2 < dt <= self.step_ms/2:
            return 0
        if note.note_type == 'don':return 2 if self.don_alt else 1
        if note.note_type == 'kat':return 4 if self.kat_alt else 3
        return 5 if note.note_type == 'don_big' else 6

    def get_state(self, lookahead=4):
        notes = [(n.time_ms-self.current_time_ms,n.note_type)
                 for n in self.hit_notes[self.next_note_idx:self.next_note_idx+lookahead]]
        notes += [(99999,'none')] * (lookahead-len(notes))
        return {'upcoming_notes':notes, 'don_alt':self.don_alt, 'kat_alt':self.kat_alt,
                'current_time_ms':self.current_time_ms, 'combo':self.combo}

    def advance(self, delta_ms):
        if delta_ms < 0: raise ValueError('Clock cannot go backwards')
        self.current_time_ms += delta_ms
        self.expire()


def metrics(games):
    events = [e for g in games for e in g.events]
    result = {key:sum(e['judgment']==key for e in events) for key in ('great','good','miss')}
    total = len(events)
    errors = np.array([e['timing_error_ms'] for e in events if e['judgment']!='miss'],dtype=float)
    for key in ('great','good','miss'):
        result[key+'_rate'] = result[key]/max(total,1)
    result['accuracy'] = (result['great']+.5*result['good'])/max(total,1)
    result['hit_rate'] = (result['great']+result['good'])/max(total,1)
    result['total_notes'] = total
    result['false_hits'] = sum(g.false_hits for g in games)
    result['full_alt_violations'] = sum(g.alt_violations for g in games)
    for key, value in [('mean_abs_error_ms',np.mean(abs(errors)) if len(errors) else None),
                       ('mean_signed_error_ms',np.mean(errors) if len(errors) else None),
                       ('median_abs_error_ms',np.median(abs(errors)) if len(errors) else None),
                       ('p95_abs_error_ms',np.percentile(abs(errors),95) if len(errors) else None)]:
        result[key] = float(value) if value is not None else None
    result['early'] = int(sum(errors<0)); result['late'] = int(sum(errors>0))
    result['accuracy_by_type'] = {t:sum(1 if e['judgment']=='great' else .5 if e['judgment']=='good' else 0
                                      for e in events if e['note_type']==t)/
                                max(1,sum(e['note_type']==t for e in events))
                                for t in ('don','kat','don_big','kat_big')}
    result['miss_reasons'] = {r:sum(e['reason']==r for e in events)
                              for r in ('unhit','wrong_color','wrong_size','late_or_early')}
    big=[e for e in events if 'big' in e['note_type']]
    result['big_note_full_hit_rate']=sum(e['judgment']!='miss' and e.get('size_correct',False) for e in big)/max(len(big),1)
    return result


PROFILES = {
    'clean':dict(latency=0.,jitter=0.,drop=0.,noise=0.,drift=0.),
    'latency':dict(latency=16.,jitter=0.,drop=0.,noise=0.,drift=0.),
    'jitter':dict(latency=0.,jitter=3.,drop=.05,noise=0.,drift=0.),
    'combined':dict(latency=16.,jitter=3.,drop=.05,noise=2.,drift=.0001),
}


class Observation:
    def __init__(self, profile, rng):
        self.profile, self.rng, self.previous = profile, rng, None

    def read(self, runtime, game):
        now = game.current_time_ms
        if self.previous is None or self.rng.random() >= self.profile['drop']:
            state = runtime.observe(game)
            # Only already-visible notes are perturbed. Hidden notes cannot appear
            # because noise happened to reduce their timestamp.
            state['upcoming_notes'] = [(dt + self.rng.normal(0,self.profile['noise']) + now*self.profile['drift'],kind)
                                       if kind!='none' else (dt,kind) for dt,kind in state['upcoming_notes']]
            self.previous = (now, copy.deepcopy(state))
        captured, state = self.previous
        state = copy.deepcopy(state)
        age = now-captured
        state['upcoming_notes'] = [(dt-age,kind) if kind!='none' else (dt,kind)
                                   for dt,kind in state['upcoming_notes']]
        state['current_time_ms'] = now
        return state
