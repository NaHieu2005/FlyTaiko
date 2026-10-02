"""
osu! Taiko Beatmap Parser

Parses .osu files and extracts Taiko-mode notes.
Handles all note types: Don (red), Kat (blue), Big Don, Big Kat,
Drum Rolls (sliders), and Swells (spinners).

Reference: https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29
"""

import os
import re
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# The model sees a 512 px image with the hit target at x=64.  osu!taiko's
# locked 16:9 playfield places the equivalent scroll start beyond the visible
# edge: (480 * 16/9 - 160) stable-space pixels from the hit target.  Mapping
# that travel time into the 448 px model/web lane is essential: copying the
# stable-space px/ms directly makes both displays scroll 1.55x too quickly.
TAIKO_MODEL_SCROLL_SCALE = (512 - 64) / (480 * 16 / 9 - 160)


@dataclass
class TaikoNote:
    """A single Taiko note."""
    time_ms: int          # Timestamp in milliseconds
    note_type: str        # 'don', 'kat', 'don_big', 'kat_big', 'drumroll', 'swell'
    end_time_ms: int = 0  # For drumrolls/swells: end time
    tick_spacing_ms: float = 0.0
    required_hits: int = 0
    strong: bool = False
    scroll_px_per_ms: float = 0.0

    @property
    def is_don(self) -> bool:
        return self.note_type in ('don', 'don_big')

    @property
    def is_kat(self) -> bool:
        return self.note_type in ('kat', 'kat_big')

    @property
    def is_big(self) -> bool:
        return self.note_type in ('don_big', 'kat_big')

    @property
    def is_hit(self) -> bool:
        """True for notes that require a single hit (don/kat)."""
        return self.note_type in ('don', 'kat', 'don_big', 'kat_big')


@dataclass
class BeatmapMetadata:
    """Metadata from a .osu file."""
    title: str = ""
    artist: str = ""
    version: str = ""      # Difficulty name
    mode: int = 0          # 0=osu!, 1=Taiko, 2=Catch, 3=Mania
    hp_drain: float = 5.0
    overall_difficulty: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0


@dataclass
class TaikoBeatmap:
    """Parsed Taiko beatmap."""
    metadata: BeatmapMetadata
    notes: List[TaikoNote] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        if not self.notes:
            return 0
        return max(n.end_time_ms if n.end_time_ms > 0 else n.time_ms
                   for n in self.notes)

    @property
    def num_hits(self) -> int:
        return sum(1 for n in self.notes if n.is_hit)

    @property
    def don_count(self) -> int:
        return sum(1 for n in self.notes if n.is_don)

    @property
    def kat_count(self) -> int:
        return sum(1 for n in self.notes if n.is_kat)

    def summary(self) -> str:
        return (
            f"'{self.metadata.title}' [{self.metadata.version}] "
            f"| {len(self.notes)} notes ({self.don_count} don, {self.kat_count} kat) "
            f"| {self.duration_ms / 1000:.1f}s "
            f"| OD={self.metadata.overall_difficulty}"
        )


def _parse_hitsound_to_type(hit_type_bits: int, hitsound: int) -> str:
    """Determine Taiko note type from hitobject type bits and hitsound.

    Taiko note mapping:
    - Don (red):     No Whistle(2) and no Clap(8) in hitsound
    - Kat (blue):    Has Whistle(2) OR Clap(8)
    - Big variant:   Has Finish(4) in hitsound

    Object types (bitmask):
    - Bit 0 (1): Hit circle (normal note)
    - Bit 1 (2): Slider (drum roll)
    - Bit 2 (4): New combo
    - Bit 3 (8): Spinner (swell)
    """
    is_slider = bool(hit_type_bits & 2)
    is_spinner = bool(hit_type_bits & 8)

    if is_spinner:
        return 'swell'
    if is_slider:
        return 'drumroll'

    # Normal hit — determine don vs kat
    has_whistle = bool(hitsound & 2)
    has_clap = bool(hitsound & 8)
    has_finish = bool(hitsound & 4)

    if has_whistle or has_clap:
        return 'kat_big' if has_finish else 'kat'
    else:
        return 'don_big' if has_finish else 'don'


def parse_osu_file(filepath: str) -> Optional[TaikoBeatmap]:
    """Parse a single .osu file and return a TaikoBeatmap if it's Taiko mode.

    Args:
        filepath: Path to .osu file

    Returns:
        TaikoBeatmap if mode=1 (Taiko), None otherwise
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as handle:
            return parse_osu_text(handle.read())
    except OSError:
        return None


def parse_osu_text(text: str) -> Optional[TaikoBeatmap]:
    """Parse native taiko text, including timing-dependent long objects."""
    # Do not interpret standard/mania objects as native taiko during imports.
    general = re.search(r'(?ms)^\[General\]\s*\n(.*?)(?=^\[|\Z)', text)
    mode = re.search(r'(?m)^Mode\s*:\s*(\d+)', general.group(1)) if general else None
    if mode is None or int(mode.group(1)) != 1:
        return None
    metadata = BeatmapMetadata()
    notes = []
    timing_points = []
    current_section = ""

    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('//'):
            continue

        # Section headers
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1]
            continue

        # Parse General section
        if current_section == 'General':
            if line.startswith('Mode:'):
                try:
                    metadata.mode = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass

        # Parse Metadata section
        elif current_section == 'Metadata':
            if line.startswith('Title:'):
                metadata.title = line.split(':', 1)[1].strip()
            elif line.startswith('Artist:'):
                metadata.artist = line.split(':', 1)[1].strip()
            elif line.startswith('Version:'):
                metadata.version = line.split(':', 1)[1].strip()

        # Parse Difficulty section
        elif current_section == 'Difficulty':
            if line.startswith('HPDrainRate:'):
                try:
                    metadata.hp_drain = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('OverallDifficulty:'):
                try:
                    metadata.overall_difficulty = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('SliderMultiplier:'):
                try:
                    metadata.slider_multiplier = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('SliderTickRate:'):
                try:
                    metadata.slider_tick_rate = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass

        elif current_section == 'TimingPoints':
            parts = line.split(',')
            if len(parts) >= 2:
                timing_points.append((float(parts[0]), float(parts[1]),
                                      len(parts)<7 or parts[6]=='1'))

        # Parse HitObjects section
        elif current_section == 'HitObjects':
            parts = line.split(',')
            if len(parts) < 5:
                continue
            try:
                time_ms = int(parts[2])
                hit_type_bits = int(parts[3])
                hitsound = int(parts[4])
            except (ValueError, IndexError):
                continue

            note_type = _parse_hitsound_to_type(hit_type_bits, hitsound)

            end_time = 0
            tick_spacing = 0.0
            required_hits = 0
            if note_type == 'swell' and len(parts) >= 6:
                # Spinner: x,y,time,type,hitSound,endTime
                try:
                    end_time = int(parts[5].split(':')[0])
                except (ValueError, IndexError):
                    raise ValueError('Malformed spinner end time')
                od = metadata.overall_difficulty
                rate = (3 + .4*od if od <= 5 else 5 + .5*(od-5)) * 1.65
                required_hits = max(1, int((end_time-time_ms)/1000 * rate))
            elif note_type == 'drumroll' and len(parts) >= 8:
                # Native taiko slider duration, including inherited velocity.
                try:
                    repeat = int(parts[6])
                    pixel_length = float(parts[7])
                    beat_length, velocity = 500.0, 1.0
                    # Red resets first; same-time green overrides regardless
                    # of file order, as LegacyBeatmapDecoder flushPendingPoints.
                    for t, length, uninherited in sorted(timing_points,key=lambda p:(p[0],not p[2])):
                        if t > time_ms: break
                        if uninherited and length > 0:
                            beat_length, velocity = length, 1.0
                        elif not uninherited and length < 0:
                            velocity = max(.1, min(10., -100./length))
                    end_time = time_ms + int(pixel_length * repeat * beat_length /
                                             (100*metadata.slider_multiplier*velocity))
                    tick_spacing = beat_length / (3 if metadata.slider_tick_rate==3 else 4)
                except (ValueError, IndexError):
                    raise ValueError('Malformed slider duration')

            notes.append(TaikoNote(
                time_ms=time_ms,
                note_type=note_type,
                end_time_ms=end_time,
                tick_spacing_ms=tick_spacing,
                required_hits=required_hits,
                strong=bool(hitsound & 4)
            ))

    # Only return if it's Taiko mode
    if metadata.mode != 1:
        return None

    # Sort by time
    notes.sort(key=lambda n: n.time_ms)

    # osu!taiko's overlapping scroll uses a per-object timing/effect-point
    # snapshot. In stable playfield units the velocity is
    # 100 px/s * hidden taiko 1.4 * SliderMultiplier * SV * 1000/beatLength.
    # Map those units into the model's 512 px image at a fixed 16:9 reference
    # aspect, matching the user's 1920x1080 osu! view.
    # Timing (red) and effect (green) points are independent; a red point
    # changes BPM but does not discard the last inherited scroll multiplier.
    # A later green line must not teleport earlier notes.
    points=sorted(timing_points, key=lambda p:(p[0],not p[2]));cursor=0
    beat_length,velocity=500.0,1.0
    for note in notes:
        while cursor<len(points) and points[cursor][0]<=note.time_ms:
            _,length,uninherited=points[cursor];cursor+=1
            if uninherited and length>0:beat_length=length
            # EffectControlPoint's scroll-speed limits differ from legacy
            # slider-duration DifficultyControlPoint limits (.1..10 above).
            elif not uninherited:velocity=max(.01,min(10.,-100./length)) if length<0 else 1.
        note.scroll_px_per_ms=(140*metadata.slider_multiplier*velocity/beat_length
                               * TAIKO_MODEL_SCROLL_SCALE)
        if not 0<note.scroll_px_per_ms<float('inf'):
            raise ValueError('Invalid taiko scroll speed')
    return TaikoBeatmap(metadata=metadata, notes=notes)


def scan_songs_folder(songs_dir: str, max_maps: int = 100) -> List[TaikoBeatmap]:
    """Scan an osu! Songs folder for Taiko beatmaps.

    Walks through all subfolders, finds .osu files,
    and returns those that are Taiko mode.

    Args:
        songs_dir: Path to osu!/Songs folder
        max_maps: Maximum number of beatmaps to return

    Returns:
        List of parsed TaikoBeatmap objects
    """
    beatmaps = []

    if not os.path.isdir(songs_dir):
        print(f"Warning: Songs directory not found: {songs_dir}")
        return beatmaps

    for root, dirs, files in os.walk(songs_dir):
        for fname in files:
            if not fname.endswith('.osu'):
                continue

            filepath = os.path.join(root, fname)
            beatmap = parse_osu_file(filepath)
            if beatmap is not None and beatmap.notes:
                beatmaps.append(beatmap)
                if len(beatmaps) >= max_maps:
                    return beatmaps

    return beatmaps


def get_hit_windows(od: float) -> dict:
    """Taiko hit windows, matching the replay evaluator's stable OD ranges."""
    od = max(0., min(10., float(od)))
    great = math.floor(50 - 3 * od) - .5
    good = math.floor(120 - 8 * od if od <= 5 else 110 - 6 * od) - .5
    miss = math.floor(135 - 8 * od if od <= 5 else 120 - 5 * od) - .5
    return {
        'great': great,
        'good': good,
        'miss': miss,
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            bm = parse_osu_file(path)
            if bm:
                print(bm.summary())
                for n in bm.notes[:20]:
                    print(f"  {n.time_ms:>8}ms  {n.note_type}")
            else:
                print("Not a Taiko beatmap or parse error.")
        elif os.path.isdir(path):
            maps = scan_songs_folder(path, max_maps=10)
            for bm in maps:
                print(bm.summary())
    else:
        print("Usage: python parser.py <path_to_osu_file_or_songs_dir>")
