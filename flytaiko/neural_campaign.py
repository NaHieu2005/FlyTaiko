"""Neural-only A/C campaign, held-out evaluation, trace export and replay.

Every generated artifact is versioned by config and beatmap content. The
legacy raw-feature policy is never loaded into this architecture.
"""
import argparse
from dataclasses import asdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from flytaiko.neural_system import NeuralConfig, NeuralRuntime, JsonBeatmap, seed_all
from flytaiko.neural_game import Game, Observation, PROFILES, metrics
from training.legacy.train_campaign import select_short_campaign


def log(root, record):
    record = {'wall_time':time.strftime('%Y-%m-%dT%H:%M:%S'), **record}
    print(json.dumps(record), flush=True)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def save_model(path, runtime, phase, epoch, optimizer, score):
    payload = {'architecture':'neural-only-v5','phase':phase,'epoch':epoch,
               'config':asdict(runtime.cfg),'fingerprint':runtime.cfg.fingerprint(),
               'num_actions':7,'action_names':['none','don_left','don_right','kat_left','kat_right','don_big','kat_big'],
               'decoder':runtime.decoder.state_dict(),'optimizer':optimizer.state_dict(),
               'score':score, 'torch_rng':torch.get_rng_state(),
               'numpy_rng':np.random.get_state(), 'random_rng':random.getstate()}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); temporary.replace(path)


def load_model(path, runtime):
    payload = torch.load(path, map_location=runtime.device, weights_only=False)
    if payload.get('architecture') != 'neural-only-v5' or payload['fingerprint'] != runtime.cfg.fingerprint():
        raise ValueError('Checkpoint architecture/config mismatch')
    runtime.decoder.load_state_dict(payload['decoder'])
    return payload


def cache_key(row, cfg, profile):
    return hashlib.sha256(json.dumps({'map':row,'config':asdict(cfg),'profile':profile},sort_keys=True).encode()).hexdigest()


def collect(rows, directory, cfg, batch_size, profile='clean', root=None):
    directory.mkdir(parents=True, exist_ok=True)
    missing=[]
    for i,row in enumerate(rows):
        meta = directory / f'{i:04d}.json'
        if meta.exists():
            if json.loads(meta.read_text())['fingerprint'] != cache_key(row,cfg,profile):
                raise ValueError(f'Stale feature cache: {meta}')
        else:
            missing.append((i,row))
    for offset in range(0,len(missing),batch_size):
        pack=missing[offset:offset+batch_size]
        runtime=NeuralRuntime(cfg,len(pack)); games=[Game(JsonBeatmap(row),cfg.step_ms) for _,row in pack]
        observers=[Observation(PROFILES[profile],np.random.default_rng(cfg.seed+i)) for i,_ in pack]
        arrays=[]; labels=[]; sizes=[]
        for (i,row),game in zip(pack,games):
            minimum_step=4 if profile!='clean' else cfg.step_ms
            n=int(np.ceil((game.hit_notes[-1].time_ms+game.windows['miss']+2*cfg.step_ms)/minimum_step))+2
            shape=(n,len(runtime.readout)*2)
            arrays.append(np.lib.format.open_memmap(directory/f'{i:04d}.partial',mode='w+',dtype=np.float16,shape=shape))
            labels.append(np.zeros(n,dtype=np.int64)); sizes.append(n)
        frame=0; start=time.perf_counter(); active=[True]*len(pack)
        step_rng=np.random.default_rng(cfg.seed+offset+5000)
        while any(active):
            states=[obs.read(runtime,game) if alive else
                    {'upcoming_notes':[(99999,'none')]*cfg.lookahead}
                    for obs,game,alive in zip(observers,games,active)]
            delta=float(np.clip(cfg.step_ms+step_rng.normal(0,PROFILES[profile]['jitter']),4,16))
            vectors=runtime.features(states,delta).cpu().numpy().astype(np.float16)
            for b,(game,alive) in enumerate(zip(games,active)):
                if not alive: continue
                if frame >= sizes[b]:raise RuntimeError('Cache frame bound violated')
                arrays[b][frame]=vectors[b]
                # Observation at t drives neurons during [t,t+step]. The
                # supervised decision and execution are timestamped at t+step.
                game.advance(delta)
                label=game.expert()
                if profile!='clean' and not game.is_done:
                    # For a jittered clock, use the first observed frame after
                    # a note, not a symmetric fixed-step window which could
                    # miss the note entirely when successive frame gaps differ.
                    note=game.hit_notes[game.next_note_idx]
                    dt=note.time_ms-game.current_time_ms
                    label=0
                    if -delta-1e-6 < dt <= 0:
                        label=(2 if game.don_alt else 1) if note.note_type=='don' else \
                              (4 if game.kat_alt else 3) if note.note_type=='kat' else \
                              5 if note.note_type=='don_big' else 6
                labels[b][frame]=label
                game.hit(label)
                if game.is_done:
                    arrays[b].flush()
                    index,row=pack[b]
                    np.save(directory/f'{index:04d}-labels.npy',labels[b][:frame+1])
                    Path(directory/f'{index:04d}.partial').replace(directory/f'{index:04d}.npy')
                    atomic_json(directory/f'{index:04d}.json',{
                        'fingerprint':cache_key(row,cfg,profile),'frames':frame+1,
                        'title':row['metadata']['title'],'profile':profile,
                        'expert':metrics([game]),'nonzero_readout':bool(np.any(arrays[b][:frame+1]!=0))})
                    active[b]=False
                    log(root or directory,{'event':'cache_complete','map':index+1,
                                           'frames':frame+1,'profile':profile})
            frame+=1
            if frame%2000==0:
                log(root or directory,{'event':'cache_progress','maps':[i+1 for i,_ in pack],
                                       'frame':frame,'batch_fps':frame*len(pack)/(time.perf_counter()-start),
                                       'profile':profile})
        del runtime,arrays
        torch.cuda.empty_cache()


@torch.no_grad()
def rollout(rows,cfg,decoder_state,batch_size=8,profile='clean',ablation=None,trace=False,compensate=False,root=None):
    all_games=[]; all_traces=[]; elapsed=[]
    for offset in range(0,len(rows),batch_size):
        pack=rows[offset:offset+batch_size]
        runtime=NeuralRuntime(cfg,len(pack)); runtime.decoder.load_state_dict(decoder_state);runtime.decoder.eval()
        if ablation=='no_synapses':
            runtime.lif.W=torch.sparse_coo_tensor(runtime.lif.W.indices(),torch.zeros_like(runtime.lif.W.values()),runtime.lif.W.shape,device=runtime.device).coalesce()
        games=[Game(JsonBeatmap(row),cfg.step_ms) for row in pack]
        observations=[Observation(PROFILES[profile],np.random.default_rng(cfg.seed+offset+i)) for i in range(len(pack))]
        queues=[[] for _ in pack]; profile_cfg=PROFILES[profile]; frame=0
        last_predictions=[0]*len(pack)
        timing_rng=np.random.default_rng(cfg.seed+offset+10000)
        trace_frames=[]
        while not all(g.is_done for g in games):
            for game,queue in zip(games,queues):
                due=[entry for entry in queue if entry[0]<=game.current_time_ms]
                queue[:]=[entry for entry in queue if entry[0]>game.current_time_ms]
                for _,action,decided in due:
                    game.hit(action)
                    if game.actions:game.actions[-1]['model_time_ms']=decided
            states=[obs.read(runtime,game) if not game.is_done else
                    {'upcoming_notes':[(99999,'none')]*cfg.lookahead}
                    for obs,game in zip(observations,games)]
            if compensate:
                for state in states:
                    state['upcoming_notes']=[(dt-profile_cfg['latency'],kind) if kind!='none' else (dt,kind)
                                              for dt,kind in state['upcoming_notes']]
            dt=float(np.clip(cfg.step_ms+timing_rng.normal(0,profile_cfg['jitter']),4,16))
            start=time.perf_counter()
            vectors=runtime.features(states,dt)
            if ablation=='zero_lif':vectors.zero_()
            actions=runtime.decoder(vectors).argmax(1).cpu().tolist()
            if cfg.edge_triggered:
                predictions=list(actions)
                actions=[0 if action and action==previous else action
                         for action,previous in zip(actions,last_predictions)]
                last_predictions=predictions
            elapsed.append((time.perf_counter()-start)*1000)
            for game,queue in zip(games,queues):
                end=game.current_time_ms+dt
                due=sorted([entry for entry in queue if entry[0]<=end])
                queue[:]=[entry for entry in queue if entry[0]>end]
                for due_time,queued_action,decided in due:
                    game.advance(max(0,due_time-game.current_time_ms))
                    game.hit(queued_action)
                    if game.actions:game.actions[-1]['model_time_ms']=decided
                game.advance(max(0,end-game.current_time_ms))
            if trace and frame%max(1,int(50/cfg.step_ms))==0:
                trace_frames.append(runtime.trace(games[0].current_time_ms))
            for game,queue,action in zip(games,queues,actions):
                if not game.is_done and action:
                    if profile_cfg['latency']==0:
                        game.hit(action)
                    else:
                        queue.append((game.current_time_ms+profile_cfg['latency'],action,game.current_time_ms))
            frame+=1
            if root and frame%2000==0:
                log(root,{'event':'eval_progress','profile':profile,'pack':offset//batch_size+1,
                          'frame':frame,'processed_notes':sum(len(g.events) for g in games)})
        all_games+=games;all_traces+=trace_frames
        del runtime;torch.cuda.empty_cache()
    report=metrics(all_games)
    report['profile']=profile;report['compensated']=compensate
    report['inference_batch_p50_ms']=float(np.percentile(elapsed,50)) if elapsed else None
    report['inference_batch_p95_ms']=float(np.percentile(elapsed,95)) if elapsed else None
    report['maps']=[{'title':g.beatmap.metadata.title,**metrics([g])} for g in all_games]
    return report,all_games,all_traces


def score(metrics_):
    n=max(metrics_['total_notes'],1)
    return metrics_['great_rate']-metrics_['miss_rate']-.1*metrics_['false_hits']/n-.05*metrics_['full_alt_violations']/n


def fit(cache_dirs,cfg,validation,output,epochs,batch_size,phase='A',initial=None,root=None):
    root=root or output;output.mkdir(parents=True,exist_ok=True)
    runtime=NeuralRuntime(cfg);optimizer=torch.optim.Adam(runtime.decoder.parameters(),lr=1e-3 if phase=='A' else 1e-4)
    start=0;best=-1e9;stale=0
    if initial:load_model(initial,runtime)
    if (output/'last.pt').exists():
        payload=load_model(output/'last.pt',runtime);optimizer.load_state_dict(payload['optimizer'])
        start=payload['epoch'];best=payload['score']
        torch.set_rng_state(payload['torch_rng'].cpu());np.random.set_state(payload['numpy_rng']);random.setstate(payload['random_rng'])
    criterion=torch.nn.CrossEntropyLoss()
    for epoch in range(start,epochs):
        runtime.decoder.train();losses=[];hit_correct=hit_total=0
        entries=[p for d in cache_dirs for p in sorted(d.glob('*.json'))]
        random.shuffle(entries)
        for p in entries:
            meta=json.loads(p.read_text());stem=p.stem
            data=np.load(p.parent/(stem+'.npy'),mmap_mode='r')
            y=np.load(p.parent/(stem+'-labels.npy'))[:meta['frames']]
            hits=np.flatnonzero(y>0);none=np.flatnonzero(y==0)
            selected=np.concatenate((hits,np.random.choice(none,min(len(none),2*len(hits)),replace=False)))
            np.random.shuffle(selected)
            # Tiny smoke maps need more than one optimizer step per epoch.
            # Full maps retain one shuffled pass through their balanced frames.
            selected=np.tile(selected, max(1, int(np.ceil(1024/max(len(selected),1)))))
            for offset in range(0,len(selected),256):
                idx=selected[offset:offset+256]
                x=torch.as_tensor(np.array(data[idx],dtype=np.float32),device=runtime.device)
                targets=torch.as_tensor(y[idx],device=runtime.device)
                logits=runtime.decoder(x);loss=criterion(logits,targets)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(runtime.decoder.parameters(),5.);optimizer.step()
                losses.append(float(loss));mask=targets>0
                hit_correct+=int(((logits.argmax(1)==targets)&mask).sum());hit_total+=int(mask.sum())
            log(root,{'event':'train_map','phase':phase,'epoch':epoch+1,'map':stem,
                      'loss':float(np.mean(losses)) if losses else None})
        state={k:v.detach().cpu().clone() for k,v in runtime.decoder.state_dict().items()}
        report,_,_=rollout(validation,cfg,state,batch_size,root=root)
        current=score(report)
        stress=None
        if phase=='C':
            stress,_,_=rollout(validation,cfg,state,batch_size,profile='combined',compensate=True,root=root)
            current=.5*current+.5*score(stress)
        improved=current>best+1e-6
        if improved:best=current;stale=0
        else:stale+=1
        save_model(output/'last.pt',runtime,phase,epoch+1,optimizer,best)
        if improved:save_model(output/'best.pt',runtime,phase,epoch+1,optimizer,best)
        atomic_json(output/f'epoch-{epoch+1}.json',{'epoch':epoch+1,'loss':float(np.mean(losses)),
                   'expert_hit_accuracy':hit_correct/max(hit_total,1),'validation':report,
                   'stress_validation':stress,'accepted':improved})
        log(root,{'event':'epoch_complete','phase':phase,'epoch':epoch+1,
                  'loss':float(np.mean(losses)),'expert_hit_accuracy':hit_correct/max(hit_total,1),'validation':report})
        if epoch>=9 and (stale>=4 or report['miss_rate']>.9):
            log(root,{'event':'early_stop','phase':phase,'epoch':epoch+1,'reason':'plateau' if stale>=4 else 'poor_validation'})
            break
    return output/'best.pt'


def replay(checkpoint_path,row,output,batch_size=1):
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False);cfg=NeuralConfig(**payload['config'])
    report,games,trace=rollout([row],cfg,payload['decoder'],1,trace=True)
    keys=['NONE','F','J','D','K','F+J','D+K']
    actions=[{**a,'action':keys[a['action_id']]} for a in games[0].actions]
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(output,{'schema_version':3,'metadata':row['metadata'],'checkpoint':str(checkpoint_path),
                       'provenance':{'connectome':'synthetic','neurons':cfg.neurons,'policy':'neural-only',
                                     'timing':'actual execution timestamps; no target snapping'},
                       'metrics':report,'actions':actions,'neural_trace':trace})
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('runs/neural_v5'))
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--neurons',type=int,default=10000)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--skip-c',action='store_true')
    args=p.parse_args();seed_all(42);torch.set_num_threads(2)
    cfg=NeuralConfig(neurons=args.neurons);root=args.root;root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text()) != asdict(cfg):
        raise ValueError('Existing run has a different config; use a new run directory')
    atomic_json(root/'config.json',asdict(cfg))
    atomic_json(root/'source_snapshot.json',{name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in ('flytaiko/neural_campaign.py','flytaiko/neural_system.py','flytaiko/neural_game.py')})
    raw=json.load(open('kaggle_data/taiko_beatmaps.json'))
    previous=json.load(open('runs/campaign_100A_50B/split.json'))
    a,b,val=select_short_campaign(raw,previous)
    used={(r['metadata']['artist'],r['metadata']['title']) for r in a+b+val}
    candidates=[r for r in raw if len(r.get('notes',[]))>=100 and max(n['t'] for n in r['notes'])<600000
                and (r['metadata']['artist'],r['metadata']['title']) not in used]
    test=[]
    for r in candidates:
        key=(r['metadata']['artist'],r['metadata']['title'])
        if key not in used:test.append(r);used.add(key)
        if len(test)==20:break
    if args.smoke:
        def snippet(row):
            r=copy.deepcopy(row);r['notes']=r['notes'][:24]
            shift=r['notes'][0]['t']-500
            for n in r['notes']:n['t']-=shift
            return r
        a=[snippet(r) for r in a[:3]];val=[snippet(r) for r in val[:2]]
        b=[snippet(r) for r in b[:2]];test=[snippet(r) for r in test[:2]]
    atomic_json(root/'split.json',{'train':[r['metadata'] for r in a],
                'validation':[r['metadata'] for r in val],'test':[r['metadata'] for r in test],
                'C':[r['metadata'] for r in b],'max_map_minutes':10})
    log(root,{'event':'start','config':asdict(cfg),'train_maps':len(a),'validation_maps':len(val),
              'test_maps':len(test),'batch':args.batch})
    collect(a,root/'cache_A',cfg,args.batch,root=root)
    best=fit([root/'cache_A'],cfg,val,root/'phase_A',args.epochs,args.batch,root=root)
    payload=torch.load(best,map_location='cpu',weights_only=False)
    baseline,_,_=rollout(val,cfg,payload['decoder'],args.batch,root=root)
    ablation,_,_=rollout(val,cfg,payload['decoder'],args.batch,ablation='zero_lif',root=root)
    no_edges,_,_=rollout(val,cfg,payload['decoder'],args.batch,ablation='no_synapses',root=root)
    atomic_json(root/'ablation.json',{'full':baseline,'zero_lif':ablation,'no_synapses':no_edges})
    healthy=(baseline['accuracy']>=.8 and baseline['accuracy']-ablation['accuracy']>=.2
             and baseline['accuracy']-no_edges['accuracy']>=.2)
    if not healthy:
        log(root,{'event':'gate_failed','reason':'A accuracy/neural contribution gate','baseline':baseline,'ablation':ablation})
        return
    selected=best
    if not args.skip_c:
        uncompensated={name:rollout(val,cfg,payload['decoder'],args.batch,profile=name,root=root)[0]
                       for name in ('latency','jitter','combined')}
        stress_before={name:rollout(val,cfg,payload['decoder'],args.batch,profile=name,compensate=True,root=root)[0]
                       for name in ('latency','jitter','combined')}
        atomic_json(root/'stress_uncalibrated.json',uncompensated)
        atomic_json(root/'stress_before.json',stress_before)
        collect(b,root/'cache_C',cfg,args.batch,profile='combined',root=root)
        candidate=fit([root/'cache_A',root/'cache_C'],cfg,val,root/'phase_C',min(args.epochs,5),args.batch,
                      phase='C',initial=best,root=root)
        c=torch.load(candidate,map_location='cpu',weights_only=False)
        clean,_,_=rollout(val,cfg,c['decoder'],args.batch,root=root)
        stress_after={name:rollout(val,cfg,c['decoder'],args.batch,profile=name,compensate=True,root=root)[0]
                      for name in ('latency','jitter','combined')}
        accepted=clean['accuracy']>=baseline['accuracy']-.005 and sum(score(v) for v in stress_after.values())>sum(score(v) for v in stress_before.values())
        atomic_json(root/'C_decision.json',{'accepted':accepted,'clean':clean,'before':stress_before,'after':stress_after})
        if accepted:selected=candidate
    final=torch.load(selected,map_location='cpu',weights_only=False)
    unseen,_,_=rollout(test,cfg,final['decoder'],args.batch,root=root)
    atomic_json(root/'unseen_test.json',unseen)
    target=json.load(open('target_map.json'));target=target[0] if isinstance(target,list) else target
    if args.smoke:target=a[0]
    output=root/'replay.json' if args.smoke else Path('web-fly/public/demos/neural-replay.json')
    report=replay(selected,target,output)
    atomic_json(root/'result.json',{'selected_checkpoint':str(selected),'unseen':unseen,'replay':report})
    log(root,{'event':'completed','checkpoint':str(selected),'replay':str(output),'metrics':report})


if __name__=='__main__':main()
