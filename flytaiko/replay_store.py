"""Durable replay catalogue. Large binary assets remain in the file store."""
import json
import os
from pathlib import Path
import sqlite3

DATABASE = Path(os.environ.get('FLYTAIKO_REPLAY_DB', Path(__file__).resolve().parent.parent / 'runs/replays.sqlite3'))

def connect():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE, timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS replays (dataset TEXT PRIMARY KEY, label TEXT NOT NULL, status TEXT NOT NULL, metadata TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)')
    db.execute('CREATE TABLE IF NOT EXISTS archived_replays (dataset TEXT PRIMARY KEY, archived_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)')
    return db

def save(dataset, label, status, metadata):
    with connect() as db:
        if db.execute('SELECT 1 FROM archived_replays WHERE dataset=?', (dataset,)).fetchone():
            return
        db.execute('INSERT INTO replays(dataset,label,status,metadata) VALUES(?,?,?,?) ON CONFLICT(dataset) DO UPDATE SET label=excluded.label,status=excluded.status,metadata=excluded.metadata,updated_at=CURRENT_TIMESTAMP',
                   (dataset, label, status, json.dumps(metadata, ensure_ascii=False)))

def save_job(job):
    chart = job.get('chart', {})
    dataset = job.get('dataset', 'user-v25-' + job['job_id'])
    label = f"{chart.get('title', 'Uploaded map')} [{chart.get('version', '')}] · {job.get('mods', 'NM')}"
    save(dataset, label, job['status'], job)

def library():
    with connect() as db:
        return {'replays': [dict(dataset=d, label=l, status=s) for d,l,s in
                            db.execute("SELECT dataset,label,status FROM replays WHERE label NOT LIKE 'Replay Upload Smoke%' ORDER BY updated_at DESC, label COLLATE NOCASE")]}

def archive(dataset):
    """Remember removals so legacy statuses cannot repopulate the catalogue."""
    with connect() as db:
        db.execute('INSERT OR IGNORE INTO archived_replays(dataset) VALUES(?)', (dataset,))
        db.execute('DELETE FROM replays WHERE dataset=?', (dataset,))

def reconcile(public, jobs):
    """Import legacy files once at startup; never scan large demos per request."""
    for path in jobs.glob('*/status.json'):
        try:
            save_job(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError):
            continue
    for path in (public / 'demos').glob('*/manifest.json'):
        try:
            manifest = json.loads(path.read_text())
            save(path.parent.name, manifest['replays'][0]['label'], 'complete', {'manifest': str(path.relative_to(public))})
        except (OSError, ValueError, KeyError, IndexError):
            continue
