import json
import replay_store

def test_catalogue_persists_and_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_store, 'DATABASE', tmp_path / 'catalogue.sqlite3')
    job={'job_id':'123456abcdef','status':'queued','chart':{'title':'Song','version':'Taiko'},'mods':'NM'}
    replay_store.save_job(job)
    assert replay_store.library()['replays'][0]['status']=='queued'
    job['status']='complete'
    replay_store.save_job(job)
    assert replay_store.library()['replays']==[{'dataset':'user-v25-123456abcdef','label':'Song [Taiko] · NM','status':'complete'}]

def test_import_old_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_store, 'DATABASE', tmp_path / 'catalogue.sqlite3')
    demo=tmp_path/'public/demos/old-demo';demo.mkdir(parents=True)
    (demo/'manifest.json').write_text(json.dumps({'replays':[{'label':'Existing'}]}))
    jobs=tmp_path/'jobs';jobs.mkdir()
    replay_store.reconcile(tmp_path/'public',jobs)
    assert replay_store.library()['replays'][0]['label']=='Existing'

def test_archived_replay_does_not_return_on_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_store, 'DATABASE', tmp_path / 'catalogue.sqlite3')
    replay_store.save('old', 'Old replay', 'complete', {})
    replay_store.archive('old')
    replay_store.save('old', 'Old replay', 'complete', {})
    assert replay_store.library()['replays'] == []

def test_internal_smoke_recordings_are_not_in_public_library(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_store, 'DATABASE', tmp_path / 'catalogue.sqlite3')
    replay_store.save('smoke', 'Replay Upload Smoke [NM]', 'complete', {})
    replay_store.save('song', 'NOCTASTRA [NM]', 'complete', {})
    assert [r['dataset'] for r in replay_store.library()['replays']] == ['song']
