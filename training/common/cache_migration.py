"""Audited, zero-copy cache/checkpoint migration; no stale-fingerprint bypass.

Capture the old numerical/teacher implementation BEFORE editing it. Reuse only
unchanged rows with identical numerical functions, physical config and graph.
Parser/runner changes are separately recorded; original artifacts remain intact.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from flytaiko.neural_campaign import atomic_json

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint(row,config,profile):
    return hashlib.sha256(json.dumps({'row':row,'config':config,'profile':profile},sort_keys=True).encode()).hexdigest()

def numerical_signatures():
    from flytaiko.malecns_training import Engine,feature_pair,teacher_action
    from flytaiko.malecns_system import ImageRetina
    from flytaiko.measured_rate_system import MeasuredRateBrain
    from flytaiko.motor_policy import output_features
    from flytaiko.motor_modes import fast_long_teacher,mode_training_label,ModeKeyInterface
    from flytaiko.visual_taiko import GameplayPixels,VisualGame
    functions=[Engine.run,Engine.brain_for,feature_pair,teacher_action,ImageRetina.sample,
        MeasuredRateBrain.__init__,MeasuredRateBrain.advance,output_features,
        fast_long_teacher,mode_training_label,ModeKeyInterface.decide,
        GameplayPixels.background,GameplayPixels.frame,
        VisualGame.__init__,VisualGame.expire,VisualGame.hit]
    return {f.__module__+'.'+f.__qualname__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in functions}

def capture(source,output):
    config=json.loads((source/'config.json').read_text())
    for name,expected in config['source_hashes'].items():
        if digest(name)!=expected:raise ValueError('Old run no longer matches source: '+name)
    atomic_json(output,{'source':str(source.resolve()),'config_sha256':digest(source/'config.json'),
        'config':config,'numerical_signatures':numerical_signatures()})
    print(json.dumps({'event':'migration_signature_captured','source':str(source),'output':str(output)}),flush=True)

def compatible(old,new):
    ignored={'architecture','source_hashes','cache_workers'}
    return {k:v for k,v in old.items() if k not in ignored}=={k:v for k,v in new.items() if k not in ignored}

def migrate_cache(source,destination,rows,oldconfig,newconfig):
    destination.mkdir(parents=True,exist_ok=True);imported=[];rejected=[]
    for p in sorted(source.glob('*.json')):
        index=int(p.stem);meta=json.loads(p.read_text())
        if index>=len(rows) or meta['row']!=rows[index]:
            rejected.append({'map':index,'reason':'Native parsed row changed'});continue
        profile=meta['profile']
        if meta['fingerprint']!=fingerprint(meta['row'],oldconfig,profile):
            raise ValueError('Invalid original fingerprint: '+str(p))
        data=np.load(p.with_suffix('.npy'),mmap_mode='r')
        labels_path=p.with_name(p.stem+'-labels.npy');initial_path=p.with_name(p.stem+'-initial.npy')
        labels=np.load(labels_path,mmap_mode='r');initial=np.load(initial_path,mmap_mode='r')
        if (data.dtype!=np.dtype(newconfig['cache_dtype']) or data.shape[1]!=4258 or
            meta['frames']>len(data) or len(labels)!=meta['frames'] or initial.shape!=(4258,)):
            raise ValueError('Malformed cache arrays: '+str(p))
        for original in (p.with_suffix('.npy'),labels_path,initial_path):
            target=destination/original.name
            if target.exists():
                if target.resolve()!=original.resolve():raise ValueError('Unexpected migration target: '+str(target))
            else:target.symlink_to(original.resolve())
        migrated={**meta,'row':rows[index],'fingerprint':fingerprint(rows[index],newconfig,profile),
            'migration':{'original_metadata':str(p.resolve()),'original_fingerprint':meta['fingerprint'],
                'verified':'identical parsed row, numerical/teacher functions, physical config and graph',
                'original_config_sha256':hashlib.sha256(json.dumps(oldconfig,sort_keys=True).encode()).hexdigest()}}
        atomic_json(destination/p.name,migrated);imported.append(index)
    return {'imported':imported,'rejected':rejected,'source':str(source),'destination':str(destination)}

def migrate(source,root,signature_path):
    from flytaiko.malecns_training import probes,VERSION
    evidence=json.loads(signature_path.read_text());oldconfig=json.loads((source/'config.json').read_text())
    config=json.loads((root/'config.json').read_text());split=json.loads((root/'split.json').read_text())
    if evidence['source']!=str(source.resolve()) or evidence['config_sha256']!=digest(source/'config.json'):
        raise ValueError('Original migration signature does not match run')
    if evidence['numerical_signatures']!=numerical_signatures():raise ValueError('Numerical/teacher functions changed')
    if not compatible(oldconfig,config):raise ValueError('Physical configuration/graph changed')
    cache_rows={'cache_pilot':probes(split['A'][:config['pilot_songs']]),'cache_A':split['A'],'cache_C':split['C']}
    reports={name:migrate_cache(source/name,root/name,rows,oldconfig,config) for name,rows in cache_rows.items()}
    checkpoint=source/'pilot/best.pt';converted_path=root/'pilot/best.pt'
    checkpoint_report=None
    if checkpoint.exists():
        converted_path.parent.mkdir(exist_ok=True)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['config']!=oldconfig:raise ValueError('Checkpoint source/config mismatch')
        checkpoint_report={'source':str(checkpoint.resolve()),'source_sha256':digest(checkpoint),
            'original_architecture':payload['architecture'],'original_config':oldconfig,
            'scope':'Unchanged motor policy weights; corrected native data requires fresh quality gate'}
        payload['migration']=checkpoint_report;payload['architecture']=VERSION;payload['config']=config
        temporary=converted_path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(converted_path)
    report={'source':str(source.resolve()),'signature':evidence,'caches':reports,'pilot_checkpoint':checkpoint_report}
    atomic_json(root/'migration.json',report)
    print(json.dumps({'event':'migration_complete','caches':{k:len(v['imported']) for k,v in reports.items()},
        'rejected':{k:v['rejected'] for k,v in reports.items()},'pilot_checkpoint':bool(checkpoint_report)}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--signature',type=Path,required=True);p.add_argument('--capture',action='store_true')
    p.add_argument('--root',type=Path);a=p.parse_args()
    if a.capture:capture(a.source,a.signature)
    elif a.root:migrate(a.source,a.root,a.signature)
    else:p.error('--root is required for migration')

if __name__=='__main__':main()
