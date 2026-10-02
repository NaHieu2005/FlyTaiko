"""Image -> measured whole MaleCNS -> motor-only readout -> all-object taiko.

Smoke runs check wiring and export real activity, NOT generalisation claims.
Production requires native .osu/.osz sources; old hit-only JSON is rejected.
"""
import argparse
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import struct
import zlib
from zipfile import ZipFile
import numpy as np
import torch
from taiko.parser import parse_osu_text
from flytaiko.extract_beatmaps import extract_beatmap_data
from flytaiko.visual_taiko import VisualGame, GameplayPixels, beatmap_from_json, all_metrics
from flytaiko.malecns_system import MeasuredBrain
from flytaiko.neural_system import NeuralDecoder, seed_all
from flytaiko.neural_campaign import atomic_json
from flytaiko.prepare_malecns import ROOT

def log(root,event,**values):
    record={'time':time.strftime('%Y-%m-%dT%H:%M:%S'),'event':event,**values}
    print(json.dumps(record),flush=True)
    with (root/'events.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')

def rgb_png(image):
    """Serialize the exact renderer framebuffer for replay inspection."""
    def chunk(kind,data):
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    h,w=image.shape[:2]
    scan=b''.join(b'\0'+row.tobytes() for row in image)
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(scan))+chunk(b'IEND',b'')
    return 'data:image/png;base64,'+base64.b64encode(png).decode()

def read_maps(source):
    rows=[]
    for p in sorted(Path(source).rglob('*')):
        if p.suffix not in ('.osu','.osz'):continue
        if p.suffix=='.osu':texts=[(str(p),p.read_text(encoding='utf-8-sig',errors='replace'))]
        else:
            with ZipFile(p) as z:texts=[(str(p)+'!'+n,z.read(n).decode('utf-8-sig')) for n in z.namelist() if n.endswith('.osu')]
        for name,text in texts:
            bm=parse_osu_text(text)
            if bm and bm.notes and bm.duration_ms<600000:
                row=extract_beatmap_data(bm);row['source']=name
                row['source_sha256']=hashlib.sha256(text.encode()).hexdigest();rows.append(row)
    return rows

def smoke_rows(rows):
    selected=[]
    seen=set()
    for row in rows:
        for kind in ('hits','don_big','kat_big','drumroll','swell'):
            if kind in seen:continue
            candidates=[n for n in row['notes'] if (n['type'] in ('don','kat','don_big','kat_big') if kind=='hits' else n['type']==kind)]
            if not candidates:continue
            n=candidates[0];start=n['t']-500
            end=(n['t']+3000 if kind in ('hits','don_big','kat_big') else n['end_t']+100)
            notes=[dict(v) for v in row['notes'] if start<=v['t']<=end and (not v['end_t'] or v['end_t']<=end)]
            for v in notes:
                v['t']-=start
                if v['end_t']:v['end_t']-=start
            selected.append({**row,'notes':notes,'smoke_kind':kind});seen.add(kind)
    if seen!={'hits','don_big','kat_big','drumroll','swell'}:raise ValueError('Smoke requires real native examples of hits, big notes, drumroll and swell')
    return selected

def simulate(brain,rows,decoder=None,trace=False,disconnect=False):
    games=[VisualGame(beatmap_from_json(row)) for row in rows];renderer=GameplayPixels()
    features=[];labels=[];traces=[];previous=[0]*len(games);frame=0
    timings=[]
    while not all(g.is_done for g in games):
        images=np.stack([renderer.frame(g) if not g.is_done else np.zeros((96,512,3),dtype=np.uint8) for g in games])
        torch.cuda.synchronize();start=time.perf_counter()
        x=brain.advance(images,disconnect=disconnect)
        torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000)
        for g in games:
            if not g.is_done:g.advance(8)
        y=np.array([g.expert() if not g.is_done else 0 for g in games],dtype=np.int64)
        if decoder is None:
            features.append(x.cpu().numpy());labels.append(y)
            actions=y
        else:
            with torch.no_grad():actions=decoder(x).argmax(1).cpu().numpy()
        for i,g in enumerate(games):
            if g.is_done:continue
            action=int(actions[i])
            if decoder is None or (action and action!=previous[i]):g.hit(action)
            previous[i]=action
        if trace and frame%6==0:
            recorded=brain.trace(games[0].current_time_ms)
            recorded['observation_time_ms']=games[0].current_time_ms-8
            recorded['rgb_png']=rgb_png(images[0]);traces.append(recorded)
        frame+=1
        if frame%250==0:print(json.dumps({'event':'simulation_progress','frame':frame}),flush=True)
    m=all_metrics(games)
    m['inference_batch_p95_ms']=float(np.percentile(timings[2:],95));m['batch']=len(games)
    return m,games,traces,np.asarray(features),np.asarray(labels)

def reset(brain):
    brain.v.fill_(-52);brain.g.zero_();brain.refractory.zero_();brain.rates.zero_()
    brain.spikes.zero_()
    brain.cursor=0;brain.elapsed=0.
    for d in brain.delay:d.zero_()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('.'))
    p.add_argument('--root',type=Path,default=Path('runs/malecns/smoke'))
    p.add_argument('--smoke',action='store_true');p.add_argument('--epochs',type=int,default=20)
    args=p.parse_args();seed_all(42);torch.set_num_threads(2)
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    rows=read_maps(args.source)
    # Archive deduplication avoids counting public osz copies as new songs.
    rows=list({r['source_sha256']:r for r in rows}.values())
    if args.smoke:rows=smoke_rows(rows)
    else:
        songs={(r['metadata']['artist'],r['metadata']['title']) for r in rows}
        if len(songs)<170:raise ValueError(f'Need native sources for 100 A + 50 C + 20 validation songs; only {len(songs)} available. Hit-only JSON cannot restore sliders/spinners.')
        raise NotImplementedError('Full training is gated on passing the new real-connectome smoke and native dataset audit')
    calibration=json.loads(Path('runs/malecns/calibration.json').read_text())
    if calibration['status']!='motor_signal_found':raise ValueError('Motor signal gate not passed')
    selected=calibration['selected'];report=json.loads((ROOT/'graph_report.json').read_text())
    source_hash={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
                 ('flytaiko/malecns_system.py','flytaiko/visual_taiko.py','flytaiko/malecns_campaign.py','taiko/parser.py')}
    config={'architecture':'malecns-image-motor-v1','dataset':report,'gain':selected['gain'],
            'tonic':selected['tonic'],'step_ms':8,'lif_dt_ms':.5,'seed':42,
            'input':'RGB viewport only','output':'2129 measured descending/motor neurons only',
            'scope':'smoke; training-set replay, not held-out performance', 'source_hashes':source_hash}
    if (root/'maps.json').exists() and json.loads((root/'maps.json').read_text())!=rows:
        raise ValueError('Existing run uses different native examples; select a new root')
    if (root/'config.json').exists():
        previous=json.loads((root/'config.json').read_text())
        if {k:v for k,v in previous.items() if k!='source_hashes'}!={k:v for k,v in config.items() if k!='source_hashes'}:
            raise ValueError('Existing run uses different graph/dynamics; select a new root')
    atomic_json(root/'config.json',config);atomic_json(root/'maps.json',rows)
    log(root,'start',maps=len(rows),config=config)
    brain=MeasuredBrain(batch=len(rows),gain=selected['gain'],tonic=selected['tonic'])
    teacher,games,_,x,y=simulate(brain,rows)
    atomic_json(root/'teacher.json',teacher)
    if teacher['miss'] or teacher['drumroll']['coverage']!=1 or teacher['swell']['coverage']!=1:
        raise ValueError('All-object expert/evaluator gate failed')
    np.savez(root/'features.npz',features=x.astype(np.float16),labels=y)
    x=x.reshape(-1,x.shape[-1]);y=y.reshape(-1)
    decoder=NeuralDecoder(x.shape[-1],256).cuda();optimizer=torch.optim.Adam(decoder.parameters(),lr=1e-3)
    hit=np.flatnonzero(y>0);none=np.flatnonzero(y==0)
    choice=np.concatenate([hit,np.random.choice(none,min(len(none),2*len(hit)),replace=False)])
    if not len(hit):raise ValueError('Empty training labels')
    best=float('inf');start_epoch=0
    if (root/'last.pt').exists():
        previous=torch.load(root/'last.pt',map_location='cuda',weights_only=False)
        if previous['graph_manifest']!=json.loads((ROOT/'source_manifest.json').read_text()):
            raise ValueError('Checkpoint graph manifest differs from loaded data')
        decoder.load_state_dict(previous['decoder']);optimizer.load_state_dict(previous['optimizer'])
        start_epoch=previous['epoch'];torch.set_rng_state(previous['torch_rng'].cpu())
        np.random.set_state(previous['numpy_rng']);random.setstate(previous['random_rng'])
        best=torch.load(root/'best.pt',map_location='cpu',weights_only=False)['loss']
        log(root,'resumed',epoch=start_epoch)
    for epoch in range(start_epoch,args.epochs):
        decoder.train();losses=[]
        # Smoke has few frames; repeat balancing for a useful optimizer check.
        indices=np.tile(choice,max(1,int(np.ceil(4096/len(choice)))));np.random.shuffle(indices)
        for offset in range(0,len(indices),256):
            idx=indices[offset:offset+256];features=torch.as_tensor(x[idx],device='cuda')
            targets=torch.as_tensor(y[idx],device='cuda');loss=torch.nn.functional.cross_entropy(decoder(features),targets)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(decoder.parameters(),5);optimizer.step();losses.append(float(loss))
        value=float(np.mean(losses));payload={'config':config,'epoch':epoch+1,
                 'decoder':decoder.state_dict(),'optimizer':optimizer.state_dict(),'loss':value,
                 'graph_manifest':json.loads((ROOT/'source_manifest.json').read_text()),
                 'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'random_rng':random.getstate()}
        temp=root/'checkpoint.tmp';torch.save(payload,temp);temp.replace(root/'last.pt')
        if value<best:
            best=value;torch.save(payload,root/'best.tmp');(root/'best.tmp').replace(root/'best.pt')
        log(root,'epoch_complete',epoch=epoch+1,loss=value)
    payload=torch.load(root/'best.pt',map_location='cuda',weights_only=False);decoder.load_state_dict(payload['decoder']);decoder.eval()
    reset(brain);evaluation,games,_,_,_=simulate(brain,rows,decoder)
    reset(brain);ablation,_,_,_,_=simulate(brain,rows,decoder,disconnect=True)
    atomic_json(root/'evaluation.json',{'full':evaluation,'disconnected':ablation})
    # Replay all three native snippets, including long objects, with actual trace.
    keys=['NONE','F','J','D','K','F+J','D+K']
    del brain;torch.cuda.empty_cache()
    brain=MeasuredBrain(batch=1,gain=selected['gain'],tonic=selected['tonic'])
    reports=[]
    for i,row in enumerate(rows):
        reset(brain);m,games,traces,_,_=simulate(brain,[row],decoder,trace=True)
        replay={'schema_version':4,'metadata':row['metadata'],'beatmap':row,'metrics':m,
                'actions':[{**a,'action':keys[a['action_id']]} for a in games[0].actions],
                'neural_trace':traces,'provenance':config,'checkpoint':str(root/'best.pt')}
        atomic_json(root/f'replay-{i}.json',replay);reports.append(m)
    passed=(evaluation['accuracy']>=.8 and evaluation['drumroll']['coverage']>=.8 and
            evaluation['swell']['coverage']==1 and evaluation['accuracy']-ablation['accuracy']>=.2)
    atomic_json(root/'result.json',{'status':'passed' if passed else 'gate_failed',
                                  'evaluation':evaluation,'ablation':ablation,'replays':reports})
    log(root,'completed',status='passed' if passed else 'gate_failed',metrics=evaluation)

if __name__=='__main__':main()
