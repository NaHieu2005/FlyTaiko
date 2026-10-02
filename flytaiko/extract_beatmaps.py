"""
Extract Taiko Beatmap Data for Kaggle Upload

Scans the osu! Songs folder, extracts ONLY the necessary note data
from Taiko beatmaps, and saves as lightweight JSON files.

Original Songs folder: ~10-50+ GB (audio, video, images, skins...)
Extracted data:        ~1-5 MB   (just timestamps + note types)

Usage:
    python extract_beatmaps.py --songs "F:\\Games\\osu!\\Songs" --output kaggle_data
    python extract_beatmaps.py --songs "F:\\Games\\osu!\\Songs" --output kaggle_data --max 50
"""

import os
import sys
import json
import argparse
from typing import List

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taiko.parser import parse_osu_file, TaikoBeatmap


def extract_beatmap_data(beatmap: TaikoBeatmap) -> dict:
    """Extract only the essential data from a parsed beatmap.

    This strips everything down to just what the training needs:
    - Note timestamps
    - Note types (don/kat/big variants)
    - Difficulty settings (for hit window calculation)
    """
    return {
        'metadata': {
            'title': beatmap.metadata.title,
            'artist': beatmap.metadata.artist,
            'version': beatmap.metadata.version,
            'overall_difficulty': beatmap.metadata.overall_difficulty,
        },
        'stats': {
            'total_notes': len(beatmap.notes),
            'don_count': beatmap.don_count,
            'kat_count': beatmap.kat_count,
            'duration_ms': beatmap.duration_ms,
        },
        'notes': [
            {
                't': note.time_ms,
                'type': note.note_type,
                'end_t': note.end_time_ms,
                'tick_spacing_ms': note.tick_spacing_ms,
                'required_hits': note.required_hits,
                'strong': note.strong,
                'scroll_px_per_ms': note.scroll_px_per_ms,
            }
            for note in beatmap.notes
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract Taiko beatmap data for Kaggle upload'
    )
    parser.add_argument(
        '--songs', type=str, required=True,
        help='Path to osu!/Songs folder (e.g. F:\\Games\\osu!\\Songs)'
    )
    parser.add_argument(
        '--output', type=str, default='kaggle_data',
        help='Output directory for extracted data'
    )
    parser.add_argument(
        '--max', type=int, default=100,
        help='Maximum number of beatmaps to extract'
    )
    parser.add_argument(
        '--min_notes', type=int, default=20,
        help='Minimum number of hit notes to include a map'
    )
    args = parser.parse_args()

    songs_dir = args.songs
    output_dir = args.output

    if not os.path.isdir(songs_dir):
        print(f"ERROR: Songs folder not found: {songs_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning: {songs_dir}")
    print(f"Output:   {output_dir}")
    print(f"Max maps: {args.max}")
    print()

    # Scan all .osu files
    all_beatmaps = []
    scanned = 0
    skipped_not_taiko = 0
    skipped_too_few = 0

    for root, dirs, files in os.walk(songs_dir):
        for fname in files:
            if not fname.endswith('.osu'):
                continue

            scanned += 1
            filepath = os.path.join(root, fname)

            try:
                beatmap = parse_osu_file(filepath)
            except Exception:
                continue

            if beatmap is None:
                skipped_not_taiko += 1
                continue

            hit_count = sum(1 for n in beatmap.notes if n.is_hit)
            if hit_count < args.min_notes:
                skipped_too_few += 1
                continue

            all_beatmaps.append(beatmap)

            if len(all_beatmaps) >= args.max:
                break

        if len(all_beatmaps) >= args.max:
            break

    print(f"Scanned:           {scanned} .osu files")
    print(f"Skipped (not Taiko): {skipped_not_taiko}")
    print(f"Skipped (too few):   {skipped_too_few}")
    print(f"Extracted:           {len(all_beatmaps)} Taiko beatmaps")
    print()

    if not all_beatmaps:
        print("No Taiko beatmaps found!")
        sys.exit(0)

    # Extract and save
    all_data = []
    for i, bm in enumerate(all_beatmaps):
        data = extract_beatmap_data(bm)
        data['id'] = i
        all_data.append(data)

    # Save as single JSON file (compact)
    output_file = os.path.join(output_dir, 'taiko_beatmaps.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, separators=(',', ':'))

    # Also save a human-readable version
    output_file_pretty = os.path.join(output_dir, 'taiko_beatmaps_readable.json')
    with open(output_file_pretty, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # Calculate sizes
    compact_size = os.path.getsize(output_file)
    pretty_size = os.path.getsize(output_file_pretty)

    print(f"Saved to:")
    print(f"  {output_file} ({compact_size / 1024:.1f} KB)")
    print(f"  {output_file_pretty} ({pretty_size / 1024:.1f} KB)")
    print()

    # Summary table
    print(f"{'#':<4} {'Title':<40} {'Diff':<15} {'Notes':>6} {'Don':>5} {'Kat':>5} {'OD':>4}")
    print("-" * 80)
    for d in all_data:
        m = d['metadata']
        s = d['stats']
        title = m['title'][:38]
        ver = m['version'][:13]
        print(f"{d['id']:<4} {title:<40} {ver:<15} {s['total_notes']:>6} "
              f"{s['don_count']:>5} {s['kat_count']:>5} {m['overall_difficulty']:>4}")

    print()
    print(f"Upload the '{output_dir}' folder to Kaggle as a dataset.")
    print(f"Then in your notebook, load with:")
    print(f"  with open('/kaggle/input/your-dataset/taiko_beatmaps.json') as f:")
    print(f"      beatmaps_data = json.load(f)")


if __name__ == '__main__':
    main()
