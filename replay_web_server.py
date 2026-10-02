"""Static FlyTaiko site and bounded OSZ replay jobs on a separate GPU host."""
import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import uuid
from urllib.parse import urlsplit
from zipfile import ZipFile, BadZipFile

from extract_beatmaps import extract_beatmap_data
from neural_campaign import atomic_json
from taiko.parser import parse_osu_text


ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / 'web-fly/public'
JOBS = ROOT / 'runs/user_replays'
MAX_UPLOAD = 128 * 1024**2
MAX_UNPACKED = 512 * 1024**2
MAX_MEMBERS = 500
LOCK = threading.Lock()
ACTIVE = {'process': None, 'job_id': None}
IDENTIFIER = re.compile(r'^[a-f0-9]{12}$')
ALLOWED_ORIGINS = {origin.rstrip('/') for origin in
                   os.environ.get('FLYTAIKO_ALLOWED_ORIGINS', '').split(',') if origin.strip()}
MIN_FREE_BYTES = 5 * 1024**3


def chart_entries(archive):
    with ZipFile(archive) as z:
        members = z.infolist()
        if len(members) > MAX_MEMBERS or sum(i.file_size for i in members) > MAX_UNPACKED:
            raise ValueError('Archive has too many files or too much uncompressed data')
        charts = []
        for item in members:
            if not item.filename.lower().endswith('.osu') or item.is_dir():
                continue
            if item.file_size > 2 * 1024**2 or item.flag_bits & 1:
                continue
            raw = z.read(item)
            parsed = parse_osu_text(raw.decode('utf-8-sig', errors='replace'))
            if parsed is None or not parsed.notes or parsed.duration_ms > 600000:
                continue
            charts.append({'member': item.filename, 'title': parsed.metadata.title,
                           'artist': parsed.metadata.artist, 'version': parsed.metadata.version,
                           'notes': parsed.num_hits, 'duration_ms': parsed.duration_ms})
        if not charts:
            raise ValueError('No Taiko chart under 10 minutes found in this .osz')
        return charts


def job_status(job_id):
    path = JOBS / job_id / 'status.json'
    if not path.exists():
        raise FileNotFoundError('Job not found')
    result = json.loads(path.read_text())
    if result['status'] == 'simulating':
        events = JOBS / job_id / 'events.jsonl'
        if events.exists():
            for line in reversed(events.read_text().splitlines()):
                record = json.loads(line)
                if record.get('event') == 'simulation_progress':
                    result['progress_frames'] = record['frame']
                    result['total_frames_approx'] = int(result['duration_ms'] / (8 * float(result.get('clock_rate', 1))))
                    break
    return result


def replay_library():
    """Published demos plus uploaded replay jobs (including pending ones)."""
    rows = []
    for manifest_path in (PUBLIC / 'demos').glob('*/manifest.json'):
        dataset = manifest_path.parent.name
        if not re.fullmatch(r'[a-zA-Z0-9-]+', dataset):
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            label = manifest['replays'][0]['label']
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
        rows.append({'dataset': dataset, 'label': label, 'status': 'complete'})
    published = {row['dataset'] for row in rows}
    for status_path in JOBS.glob('*/status.json'):
        job_id = status_path.parent.name
        if not IDENTIFIER.fullmatch(job_id):
            continue
        if any(dataset in published for dataset in ('user-v23-' + job_id, 'user-v24-' + job_id,
                                                     'user-v25-' + job_id)):
            continue
        try:
            status = json.loads(status_path.read_text())
            if status['status'] not in ('queued', 'simulating', 'publishing'):
                continue
            chart = status['chart']
            label = f"{chart['title']} [{chart['version']}] · {status.get('mods', 'NM')} · creating"
        except (OSError, ValueError, KeyError, TypeError):
            continue
        dataset = {'v23': 'user-v23-', 'v24-purple': 'user-v24-',
                   'v25': 'user-v25-'}.get(status.get('model'), 'user-v23-') + job_id
        rows.append({'dataset': dataset, 'label': label, 'status': status['status']})
    rows.sort(key=lambda row: (not row['dataset'].startswith(('user-v23-', 'user-v24-', 'user-v25-')),
                               row['label'].casefold()))
    return {'replays': rows}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def allowed_origin(self):
        origin = self.headers.get('Origin', '').rstrip('/')
        if origin in ALLOWED_ORIGINS:
            return origin
        # The localhost workflow uses a same-origin frontend and API.
        host = self.headers.get('Host', '')
        if origin and urlsplit(origin).netloc == host and urlsplit(origin).scheme in ('http', 'https'):
            return origin
        return None

    def end_headers(self):
        origin = self.allowed_origin()
        if origin:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        super().end_headers()

    def do_OPTIONS(self):
        if not self.allowed_origin() or not self.path.startswith('/api/'):
            self.respond(403, {'error': 'Origin not allowed'})
            return
        self.send_response(204)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '600')
        self.end_headers()

    def respond(self, code, body):
        payload = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/api/replays':
            self.respond(200, replay_library())
            return
        match = re.fullmatch(r'/api/replays/([a-f0-9]{12})', self.path)
        if match:
            try:
                self.respond(200, job_status(match.group(1)))
            except FileNotFoundError:
                self.respond(404, {'error': 'Job not found'})
            return
        return super().do_GET()

    def do_POST(self):
        try:
            if self.headers.get('Origin') and not self.allowed_origin():
                return self.respond(403, {'error': 'Origin not allowed'})
            length = int(self.headers.get('Content-Length', '-1'))
            if length < 1 or length > MAX_UPLOAD:
                return self.respond(413, {'error': 'Upload must be 1 byte to 128 MB'})
            if self.path == '/api/osz':
                if shutil.disk_usage(JOBS).free < MIN_FREE_BYTES + length:
                    return self.respond(507, {'error': 'Insufficient disk space for another upload'})
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/octet-stream':
                    return self.respond(415, {'error': 'Expected raw .osz file'})
                upload_id = uuid.uuid4().hex[:12]
                folder = JOBS / upload_id
                folder.mkdir(parents=True, exist_ok=False)
                archive = folder / 'uploaded.osz'
                with archive.open('wb') as target:
                    remaining = length
                    while remaining:
                        data = self.rfile.read(min(1024 * 1024, remaining))
                        if not data:
                            raise ValueError('Upload ended early')
                        target.write(data)
                        remaining -= len(data)
                charts = chart_entries(archive)
                atomic_json(folder / 'upload.json', {'charts': charts,
                                                     'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
                return self.respond(200, {'upload_id': upload_id, 'charts': charts})
            if self.path == '/api/replays':
                if shutil.disk_usage(JOBS).free < MIN_FREE_BYTES:
                    return self.respond(507, {'error': 'Insufficient disk space for another replay'})
                if length > 4096:
                    return self.respond(413, {'error': 'Request too large'})
                data = json.loads(self.rfile.read(length))
                upload_id = data.get('upload_id')
                chart_index = data.get('chart_index')
                mods = data.get('mods', 'NM')
                model = data.get('model', 'v25')
                if mods not in ('NM', 'HR', 'DT', 'DTHR'):
                    raise ValueError('Unsupported mods')
                if model != 'v25':
                    raise ValueError('Unsupported model')
                if not isinstance(upload_id, str) or not IDENTIFIER.fullmatch(upload_id):
                    raise ValueError('Invalid upload identifier')
                folder = JOBS / upload_id
                upload = json.loads((folder / 'upload.json').read_text())
                if not isinstance(chart_index, int) or not 0 <= chart_index < len(upload['charts']):
                    raise ValueError('Invalid chart selection')
                with LOCK:
                    if ACTIVE['process'] is not None and ACTIVE['process'].poll() is None:
                        return self.respond(409, {'error': 'A replay job is already running'})
                    job_id = uuid.uuid4().hex[:12]
                    job_dir = JOBS / job_id
                    job_dir.mkdir(parents=True, exist_ok=False)
                    atomic_json(job_dir / 'status.json', {'status': 'queued', 'job_id': job_id,
                             'chart': upload['charts'][chart_index], 'upload_id': upload_id,
                             'mods': mods, 'model': model})
                    log = (job_dir / 'worker.log').open('w')
                    command = [sys.executable, '-u', 'replay_osz_worker.py',
                               '--upload', upload_id, '--chart', str(chart_index), '--job', job_id,
                               '--mods', mods, '--model', model]
                    try:
                        process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                                   stderr=subprocess.STDOUT)
                    finally:
                        log.close()
                    ACTIVE.update(process=process, job_id=job_id)
                return self.respond(202, {'job_id': job_id, 'status_url': f'/api/replays/{job_id}'})
            return self.respond(404, {'error': 'Unknown API route'})
        except (ValueError, KeyError, BadZipFile, json.JSONDecodeError) as error:
            return self.respond(400, {'error': str(error)[:300]})
        except FileNotFoundError:
            return self.respond(404, {'error': 'Upload not found'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--bind', default='0.0.0.0')
    args = parser.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f'FlyTaiko replay server listening on {args.bind}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
