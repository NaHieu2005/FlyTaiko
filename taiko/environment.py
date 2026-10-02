"""
Taiko Game Environment

Simulates an osu! Taiko gameplay session from a parsed beatmap.
Provides game state for the fly brain and processes hit actions.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .parser import TaikoBeatmap, TaikoNote, get_hit_windows


# Canonical 7-action space for realtime Taiko play.
ACTION_NONE = 0
ACTION_DON_LEFT = 1      # Red center hit, left/primary Don key
ACTION_DON_RIGHT = 2     # Red center hit, right/alternate Don key
ACTION_KAT_LEFT = 3      # Blue rim hit, left/primary Kat key
ACTION_KAT_RIGHT = 4     # Blue rim hit, right/alternate Kat key
ACTION_DON_BIG = 5       # Big red (both Don keys)
ACTION_KAT_BIG = 6       # Big blue (both Kat keys)
NUM_ACTIONS = 7

ACTION_DON = ACTION_DON_LEFT
ACTION_KAT = ACTION_KAT_LEFT

ACTION_NAMES = [
    'none', 'don_left', 'don_right', 'kat_left', 'kat_right',
    'don_big', 'kat_big'
]
DON_ACTIONS = {ACTION_DON_LEFT, ACTION_DON_RIGHT, ACTION_DON_BIG}
KAT_ACTIONS = {ACTION_KAT_LEFT, ACTION_KAT_RIGHT, ACTION_KAT_BIG}
BIG_ACTIONS = {ACTION_DON_BIG, ACTION_KAT_BIG}


@dataclass
class HitResult:
    """Result of a hit attempt."""
    judgment: str        # 'great', 'good', 'miss', 'ignore'
    reward: float        # Reward signal for the brain
    note_type: str       # What the correct note was
    action: int          # What action was taken
    timing_error_ms: float = 0.0  # How far off the timing was


class TaikoEnvironment:
    """Simulates osu! Taiko gameplay for training.

    The environment steps through time and provides game state
    to the brain. The brain outputs actions, and the environment
    returns rewards based on hit accuracy.
    """

    def __init__(self, beatmap: TaikoBeatmap, step_ms: int = 16):
        """
        Args:
            beatmap: Parsed TaikoBeatmap
            step_ms: Time step in ms (default ~60fps)
        """
        self.beatmap = beatmap
        self.step_ms = step_ms

        # Hit windows based on OD
        self.windows = get_hit_windows(beatmap.metadata.overall_difficulty)

        # Filter to hittable notes only (don, kat, big variants)
        self.hit_notes = [n for n in beatmap.notes if n.is_hit]

        # State
        self.current_time_ms = 0
        self.next_note_idx = 0
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.greats = 0
        self.goods = 0
        self.misses = 0
        self.total_hits = 0

        # Alternation state: False means the next same-color single hit uses
        # the left/primary key, True means it uses the right/alternate key.
        self.don_alt = False
        self.kat_alt = False

        # History for analysis
        self.history: List[HitResult] = []

    def reset(self):
        """Reset environment for a new episode."""
        self.current_time_ms = 0
        self.next_note_idx = 0
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.greats = 0
        self.goods = 0
        self.misses = 0
        self.total_hits = 0
        self.don_alt = False
        self.kat_alt = False
        self.history = []

    @property
    def is_done(self) -> bool:
        """True when all notes have passed."""
        if not self.hit_notes:
            return True
        last_note_time = self.hit_notes[-1].time_ms
        return (self.current_time_ms > last_note_time + self.windows['miss'] + 100
                and self.next_note_idx >= len(self.hit_notes))

    def get_state(self, lookahead: int = 8) -> dict:
        """Get current game state for the brain.

        Args:
            lookahead: Number of upcoming notes to include

        Returns:
            Dictionary with:
            - upcoming_notes: List of (time_until_ms, note_type_str)
            - combo: current combo
            - last_judgment: last hit result
            - progress: fraction of song completed (0-1)
        """
        upcoming = []
        for i in range(self.next_note_idx,
                       min(self.next_note_idx + lookahead, len(self.hit_notes))):
            note = self.hit_notes[i]
            time_until = note.time_ms - self.current_time_ms
            upcoming.append((time_until, note.note_type))

        # Pad with dummy far-away notes if fewer than lookahead
        while len(upcoming) < lookahead:
            upcoming.append((99999, 'none'))

        last_judgment = self.history[-1].judgment if self.history else 'none'

        progress = 0.0
        if self.hit_notes:
            progress = self.current_time_ms / (self.hit_notes[-1].time_ms + 1000)
            progress = min(max(progress, 0.0), 1.0)

        return {
            'upcoming_notes': upcoming,
            'combo': self.combo,
            'last_judgment': last_judgment,
            'progress': progress,
            'current_time_ms': self.current_time_ms,
            'don_alt': self.don_alt,
            'kat_alt': self.kat_alt,
        }

    def get_expert_action(self) -> int:
        """Get the perfect action for the current timestep.

        Returns the action that the expert (perfect) player would take.
        """
        if self.next_note_idx >= len(self.hit_notes):
            return ACTION_NONE

        note = self.hit_notes[self.next_note_idx]
        time_until = note.time_ms - self.current_time_ms

        # Imitation target: choose the closest simulation frame to the note.
        if abs(time_until) <= self.step_ms / 2:
            if note.note_type == 'don':
                return ACTION_DON_RIGHT if self.don_alt else ACTION_DON_LEFT
            elif note.note_type == 'kat':
                return ACTION_KAT_RIGHT if self.kat_alt else ACTION_KAT_LEFT
            elif note.note_type == 'don_big':
                return ACTION_DON_BIG
            elif note.note_type == 'kat_big':
                return ACTION_KAT_BIG

        return ACTION_NONE

    def step(self, action: int) -> HitResult:
        """Process one timestep with the given action.

        Args:
            action: One of ACTION_* constants

        Returns:
            HitResult with judgment and reward
        """
        result = HitResult(
            judgment='ignore',
            reward=0.0,
            note_type='none',
            action=action,
        )

        # Check for missed notes (passed without being hit)
        self._check_missed_notes()

        # Process action
        if action != ACTION_NONE and self.next_note_idx < len(self.hit_notes):
            note = self.hit_notes[self.next_note_idx]
            timing_error = abs(note.time_ms - self.current_time_ms)
            result.note_type = note.note_type
            result.timing_error_ms = timing_error

            if action == ACTION_DON_LEFT:
                self.don_alt = True
            elif action == ACTION_DON_RIGHT:
                self.don_alt = False
            elif action == ACTION_KAT_LEFT:
                self.kat_alt = True
            elif action == ACTION_KAT_RIGHT:
                self.kat_alt = False

            # Check if within any hit window
            if timing_error <= self.windows['miss']:
                # Check if correct type
                action_is_don = action in DON_ACTIONS
                note_is_don = note.is_don

                if action_is_don == note_is_don:
                    # Correct type! Determine judgment
                    if timing_error <= self.windows['great']:
                        result.judgment = 'great'
                        result.reward = 1.0
                        self.score += 300
                        self.greats += 1
                    elif timing_error <= self.windows['good']:
                        result.judgment = 'good'
                        result.reward = 0.5
                        self.score += 150
                        self.goods += 1
                    else:
                        result.judgment = 'miss'
                        result.reward = -0.5
                        self.misses += 1

                    # Big note bonus
                    action_is_big = action in BIG_ACTIONS
                    if note.is_big and action_is_big and result.judgment != 'miss':
                        result.reward += 0.2
                        self.score += 100

                    # Combo
                    if result.judgment in ('great', 'good'):
                        self.combo += 1
                        self.max_combo = max(self.max_combo, self.combo)
                        result.reward += min(self.combo * 0.02, 0.5)  # Combo bonus
                    else:
                        self.combo = 0

                    self.total_hits += 1
                    self.next_note_idx += 1
                else:
                    # Wrong type (don when should kat, etc.)
                    result.judgment = 'miss'
                    result.reward = -1.0
                    self.combo = 0
                    self.misses += 1
                    self.total_hits += 1
                    self.next_note_idx += 1

        self.history.append(result)

        # Advance time
        self.current_time_ms += self.step_ms

        return result

    def _check_missed_notes(self):
        """Mark notes as missed if they've passed the miss window."""
        while self.next_note_idx < len(self.hit_notes):
            note = self.hit_notes[self.next_note_idx]
            if self.current_time_ms > note.time_ms + self.windows['miss']:
                # Missed this note
                miss_result = HitResult(
                    judgment='miss',
                    reward=-1.0,
                    note_type=note.note_type,
                    action=ACTION_NONE,
                )
                self.history.append(miss_result)
                self.combo = 0
                self.misses += 1
                self.total_hits += 1
                self.next_note_idx += 1
            else:
                break

    def get_results_summary(self) -> dict:
        """Get summary of the episode."""
        total = max(self.total_hits, 1)
        return {
            'score': self.score,
            'max_combo': self.max_combo,
            'greats': self.greats,
            'goods': self.goods,
            'misses': self.misses,
            'total': self.total_hits,
            'accuracy': (self.greats + self.goods) / total,
            'great_rate': self.greats / total,
        }

    def iterate_timesteps(self):
        """Generator that yields timesteps until the song ends."""
        while not self.is_done:
            yield self.current_time_ms
