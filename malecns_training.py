"""Resumable native-data A/C campaign with measured image-to-motor connectome.

No live policy sees labels, note timestamps, object types or the game clock.
Teachers and evaluators may read those; retinal input is only framebuffer RGB.
"""
import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from neural_campaign import atomic_json
from neural_system import seed_all
from malecns_system import MeasuredBrain
from malecns_campaign import reset, rgb_png, log
from motor_policy import MotorPolicy, KeyInterface, ACTION_TO_CATEGORY, output_features
from visual_taiko import VisualGame, GameplayPixels, beatmap_from_json, all_metrics
from prepare_malecns import ROOT

VERSION='malecns-image-motor-v18-osu-taiko-16x9-sv'
PROFILES={'clean':dict(latency=0,jitter=0,drop=0,noise=0,drift=0),
          'latency':dict(latency=16,jitter=0,drop=0,noise=0,drift=0),
          'jitter':dict(latency=0,jitter=2,drop=.05,noise=0,drift=0),
          'combined':dict(latency=16,jitter=2,drop=.05,noise=2,drift=.0001)}

def song_key(row):
    m=row['metadata']
    return (m['artist'].strip().casefold(),m['title'].strip().casefold())

def endpoint(row):
    return max(max(n['t'],n.get('end_t',0)) for n in row['notes'])

def choose_split(rows,counts=(100,50,20,20),seed=42):
    groups={};excluded=[]
    for row in rows:
        try:bm=beatmap_from_json(row)
        except (ValueError,TypeError) as e:
            excluded.append({'source':row.get('source'),'reason':str(e)});continue
        hits=[n['t'] for n in row['notes'] if n['type'] in ('don','kat','don_big','kat_big')]
        if len(hits)<100 or not 20000<endpoint(row)<600000 or min(hits)<0:continue
        # A one-press-per-decision interface cannot represent simultaneous opposite
        # colors; select representable native charts BEFORE model evaluation.
        if len(hits)>1 and min(np.diff(sorted(hits)))<16:
            excluded.append({'source':row.get('source'),'reason':'Circle spacing <16ms: 8ms decision plus release frame'});continue
        key=song_key(row);types={n.note_type for n in bm.notes}
        # Prefer modest-density charts and complete object coverage, not solely
        # the hardest difficulty of every song.
        quality=2*('drumroll' in types)+2*('swell' in types)+('don_big' in types)+('kat_big' in types)-len(hits)/max(endpoint(row)/1000,1)/20
        if key not in groups or quality>groups[key][0]:groups[key]=(quality,row)
    rng=random.Random(seed);candidates=[v[1] for v in groups.values()];rng.shuffle(candidates)
    if len(candidates)<sum(counts):raise ValueError(f'Only {len(candidates)} eligible independent songs')
    result={};offset=0
    for name,count in zip(('A','C','validation','test'),counts):
        result[name]=candidates[offset:offset+count];offset+=count
    keys=[{song_key(r) for r in result[k]} for k in result]
    if any(a&b for i,a in enumerate(keys) for b in keys[i+1:]):raise RuntimeError('Song leakage')
    for name in ('A','C'):
        present={n['type'] for r in result[name] for n in r['notes']}
        if not {'don','kat','don_big','kat_big','drumroll','swell'}<=present:
            raise ValueError(f'{name} lacks full object coverage')
    return result,excluded

def excerpt(row,centre_ms,length_ms=8000):
    start=max(0,centre_ms-1000);end=start+length_ms
    notes=[copy.deepcopy(n) for n in row['notes'] if start<=n['t']<end and
           (not n.get('end_t') or n['end_t']<end)]
    if not notes:return None
    for n in notes:
        n['t']-=start
        if n.get('end_t'):n['end_t']-=start
    return {**row,'notes':notes,'excerpt_start_ms':start,'excerpt_length_ms':length_ms}

def probes(rows,length=8000):
    result=[]
    for row in rows:
        hits=[n for n in row['notes'] if n['type'] in ('don','kat','don_big','kat_big')]
        anchors=[hits[len(hits)//3]['t']]
        longs=[n for n in row['notes'] if n['type'] in ('drumroll','swell') and n['end_t']-n['t']<length-2000]
        if longs:anchors.append(longs[0]['t'])
        else:anchors.append(hits[2*len(hits)//3]['t'])
        for t in anchors:
            piece=excerpt(row,t,length)
            if piece:result.append(piece)
    return result

def pilot_validation(rows,count=8):
    """Deterministic object coverage, selected before seeing model quality."""
    candidates=probes(rows);selected=[];covered=set();used=set()
    while len(selected)<count and candidates:
        available=[r for r in candidates if song_key(r) not in used] or candidates
        row=max(available,key=lambda r:len({n['type'] for n in r['notes']}-covered))
        selected.append(row);covered.update(n['type'] for n in row['notes']);used.add(song_key(row))
        candidates.remove(row)
    if not {'don','kat','don_big','kat_big','drumroll','swell'}<=covered:
        raise ValueError('Pilot validation lacks native object classes')
    return selected

def teacher_action(game):
    """Training-only first-frame-at/after-target labels work with variable steps."""
    t=game.current_time_ms
    if game.next_note_idx<len(game.hit_notes):
        note=game.hit_notes[game.next_note_idx]
        if 0<=t-note.time_ms<=game.windows['great']:
            if note.note_type=='don':return 2 if game.don_alt else 1
            if note.note_type=='kat':return 4 if game.kat_alt else 3
            return 5 if note.note_type=='don_big' else 6
    for obj in game.long:
        n=obj['note']
        if obj['finished'] or t<n.time_ms:continue
        if n.note_type=='drumroll':
            half=n.tick_spacing_ms/2
            # Rounded final ticks can land at the object's expiry boundary.
            # Press on the last valid decision before expiry, not after it.
            if any(i not in obj['hit_ticks'] and (
                0<=t-v<=half or (0<v-t<=half and
                t+game.step_ms>min(v+half,n.end_time_ms+half)))
                for i,v in enumerate(obj['ticks'])):
                return 2 if game.don_alt else 1
        elif t<=n.end_time_ms and obj['hits']<n.required_hits:
            # Finish an earlier spinner before a nested long object starts.
            # This is teacher scheduling only; the live policy still sees RGB.
            following=[v['note'].time_ms for v in game.long
                       if n.time_ms<v['note'].time_ms<n.end_time_ms]
            cutoff=min(following,default=n.end_time_ms)
            duration=min(n.end_time_ms-n.time_ms,
                         max(16*n.required_hits,cutoff-n.time_ms-2*game.step_ms))
            if t>=n.time_ms+(obj['hits']+.5)*duration/n.required_hits:
                return (2 if game.don_alt else 1) if obj['hits']%2==0 else (4 if game.kat_alt else 3)
    return 0

def feature_pair(current,previous):
    # Relative membrane/rate change uses motor history only, never game state.
    return torch.cat((current,(current-previous).clamp(-2,2)),dim=1)

def audit_expert_split(split,root,fast=False):
    records=[]
    for phase,rows in split.items():
        for i,row in enumerate(rows):
            game=VisualGame(beatmap_from_json(row))
            while not game.is_done:
                game.advance(8)
                action=teacher_action(game)
                if fast:
                    from motor_modes import fast_long_teacher
                    action=fast_long_teacher(game,teacher_action)
                game.hit(action)
            result=all_metrics([game])
            passed=(result['accuracy']>=.98 and result['false_hits']==0 and
                    result['full_alt_violations']==0 and all(
                        not result[k]['required'] or result[k]['coverage']==1 for k in ('drumroll','swell')))
            records.append({'phase':phase,'index':i,'source':row['source'],'passed':passed,'metrics':result})
            if (i+1)%10==0:log(root,'expert_preflight_progress',phase=phase,maps=i+1)
    atomic_json(root/'expert_preflight.json',records)
    if not all(r['passed'] for r in records):
        raise RuntimeError('Native expert/evaluator preflight failed; see expert_preflight.json')
    log(root,'expert_preflight_passed',maps=len(records))

class Engine:
    def __init__(self,config,root):
        self.config,self.root=config,root;self.brain=None;self.sensor=None
        style=config.get('observation_style','legacy')
        if style=='legacy':self.renderer=GameplayPixels()
        elif style=='web-default':
            from visual_taiko import DefaultSkinGameplayPixels
            self.renderer=DefaultSkinGameplayPixels()
        elif style=='web-native-resolution':
            from highres_taiko import HighResolutionTaikoPixels
            self.renderer=HighResolutionTaikoPixels()
        elif style=='web-native-purple-spinner':
            from highres_taiko import HighResolutionTaikoPixels
            self.renderer=HighResolutionTaikoPixels(spinner_palette='purple')
        else:raise ValueError('Unknown observation style: '+str(style))

    def brain_for(self,batch):
        if self.brain is None or self.brain.batch!=batch:
            self.brain=None;torch.cuda.empty_cache()
            if self.config.get('neuron_model')=='graded-rate engineering hypothesis':
                from measured_rate_system import MeasuredRateBrain
                self.brain=MeasuredRateBrain(batch=batch,coupling=self.config.get('rate_coupling',.95),baseline=.5)
            else:self.brain=MeasuredBrain(batch=batch,gain=self.config['gain'],tonic=self.config['tonic'],
                                    tonic_scope=self.config.get('tonic_scope','all'),
                                    body_tonic=self.config.get('body_tonic',0.),photo_tonic=self.config.get('photo_tonic'))
        reset(self.brain);return self.brain

    def run(self,rows,policy=None,profile='clean',threshold=.65,disconnect=False,
            collect_callback=None,trace=False,seed=42,training_label_fn=None):
        brain=self.brain_for(len(rows));games=[VisualGame(beatmap_from_json(r)) for r in rows]
        if self.config.get('decoder_input')=='sensory_memory' and self.sensor is None:
            from sensory_readout import SensoryReadout
            self.sensor=SensoryReadout(brain.metadata,brain.arrays)
        brain.retina.bounds=tuple(self.config.get('retina_bounds',(0.,0.,1.,1.)))
        brain.retina.bilinear=self.config.get('retina_bilinear',False)
        brain.retina.x_warp=tuple(self.config['retina_x_warp']) if self.config.get('retina_x_warp') else None
        settings=PROFILES[profile];rng=np.random.default_rng(seed)
        clock_rate=float(self.config.get('clock_rate',1.))
        if not 0.5<=clock_rate<=2.0:raise ValueError('Unsafe gameplay clock rate')
        for game in games:game.current_time_ms=-8.*clock_rate
        shape=(self.renderer.height,self.renderer.width,3)
        queues=[deque() for _ in rows];held=[np.zeros(shape,dtype=np.uint8) for _ in rows]
        interface=KeyInterface
        if self.config.get('motor_interface')=='continuous_modes':
            from motor_modes import ModeKeyInterface
            interface=ModeKeyInterface
        keys=[interface(threshold=threshold,release=min(.35,threshold-.05)) for _ in rows]
        if self.config.get('warmup_ms',0):
            brain.advance(np.stack([self.renderer.background()]*len(rows)),duration=self.config['warmup_ms'],disconnect=disconnect)
        previous=output_features(brain).float()
        if self.config.get('cache_dtype','float16')=='float16':previous=previous.half().float()
        self.initial_motor=previous.cpu().numpy().copy()
        lag=int(self.config.get('motor_delta_lag_frames',1))
        if lag<1:raise ValueError('motor_delta_lag_frames must be positive')
        motor_history=deque([previous]*lag,maxlen=lag)
        if policy is not None and hasattr(policy,'reset_state'):policy.reset_state()
        traces=[];timings=[];frame=0;model_time=0.
        while not all(g.is_done for g in games):
            delta=8. if not settings['jitter'] else float(np.clip(round((8+rng.normal(0,settings['jitter']))*2)/2,4,12))
            frames=[];was_done=[g.is_done for g in games]
            for i,g in enumerate(games):
                image=self.renderer.frame(g) if not g.is_done else np.zeros(shape,dtype=np.uint8)
                queues[i].append((model_time,image));target=min(model_time,model_time*(1+settings['drift'])-settings['latency'])
                while len(queues[i])>1 and queues[i][1][0]<=target:queues[i].popleft()
                observed=queues[i][0][1] if queues[i][0][0]<=target else np.zeros_like(image)
                if rng.random()>=settings['drop']:held[i]=observed
                pixels=held[i]
                if settings['noise']:pixels=np.clip(pixels.astype(float)+rng.normal(0,settings['noise'],pixels.shape),0,255).astype(np.uint8)
                frames.append(pixels)
            images=np.stack(frames)
            torch.cuda.synchronize();start=time.perf_counter();brain.advance(images,duration=delta,disconnect=disconnect)
            # Cache and live readout use exactly the same quantization, so the
            # temporal differences match storage/training bit for bit.
            current=output_features(brain).float()
            if self.config.get('cache_dtype','float16')=='float16':current=current.half().float()
            features=(self.sensor.features(brain,current,motor_history[0]) if self.sensor is not None
                      else feature_pair(current,motor_history[0]))
            motor_history.append(current)
            if not torch.isfinite(features).all():raise RuntimeError('Non-finite neural state')
            if policy is not None:
                with torch.no_grad():probabilities=torch.softmax(policy(features),dim=1).cpu().numpy()
            torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000)
            for g in games:
                if not g.is_done:g.step_ms=delta*clock_rate;g.advance(delta*clock_rate)
            teacher=teacher_action
            if self.config.get('motor_interface')=='continuous_modes':
                from motor_modes import fast_long_teacher,mode_training_label
                teacher=lambda g:fast_long_teacher(g,teacher_action)
            expert=(np.array([teacher(g) if not g.is_done else 0 for g in games],dtype=np.int64)
                    if policy is None else np.zeros(len(games),dtype=np.int64))
            if collect_callback is not None:
                labels=ACTION_TO_CATEGORY[expert]
                if self.config.get('motor_interface')=='continuous_modes':
                    label_fn=training_label_fn or mode_training_label
                    labels=np.array([label_fn(g) if not g.is_done else 0 for g in games],dtype=np.int8)
                cached=features if self.sensor is not None else current
                collect_callback(frame,cached.cpu().numpy(),labels,was_done)
            for i,g in enumerate(games):
                if g.is_done:continue
                action=int(expert[i]) if policy is None else keys[i].decide(probabilities[i],delta_ms=delta)
                if action:g.hit(action)
            if trace and frame%6==0:
                record=brain.trace(games[0].current_time_ms);record['observation_time_ms']=games[0].current_time_ms-delta*clock_rate
                record['rgb_png']=rgb_png(images[0]);traces.append(record)
            frame+=1;model_time+=delta
            if frame%2000==0:
                log(self.root,'simulation_progress',profile=profile,frame=frame,
                    maps=[r['metadata']['title'] for r in rows],gpu_mb=torch.cuda.memory_allocated()/1024**2)
            if frame>int(max(endpoint(r) for r in rows)/4)+1000:raise RuntimeError('Timeline watchdog exceeded map duration')
        report=all_metrics(games);report.update(profile=profile,inference_batch_p50_ms=float(np.median(timings[2:] or timings)),
                    inference_batch_p95_ms=float(np.percentile(timings[2:] or timings,95)),batch=len(rows))
        return report,games,traces

    def cache(self,rows,directory,profile='clean',indices=None):
        directory.mkdir(parents=True,exist_ok=True);missing=[]
        selected=list(range(len(rows))) if indices is None else list(indices)
        if len(selected)!=len(set(selected)):raise ValueError('Duplicate cache indices')
        if any(i<0 or i>=len(rows) for i in selected):raise ValueError('Cache index outside rows')
        for i in selected:
            row=rows[i]
            fingerprint=hashlib.sha256(json.dumps({'row':row,'config':self.config,'profile':profile},sort_keys=True).encode()).hexdigest()
            meta=directory/f'{i:04d}.json'
            if meta.exists():
                if json.loads(meta.read_text())['fingerprint']!=fingerprint:raise ValueError('Stale cache '+str(meta))
            else:missing.append((i,row,fingerprint))
        batch=self.config['batch']
        for offset in range(0,len(missing),batch):
            if shutil.disk_usage(directory).free<20*1024**3:raise RuntimeError('Disk safety reserve <20GB; preserving existing cache')
            pack=missing[offset:offset+batch];arrays=[];labels=[];sizes=[0]*len(pack)
            minimum=4 if profile!='clean' else 8
            arrays_info=np.load(ROOT/'graph_arrays.npz')
            width=(len(arrays_info['photo'])+1024+4*len(arrays_info['output'])
                   if self.config.get('decoder_input')=='sensory_memory'
                   else len(arrays_info['output'])*2)
            for i,row,_ in pack:
                n=int(endpoint(row)/minimum)+1000
                arrays.append(np.lib.format.open_memmap(directory/f'{i:04d}.partial',mode='w+',
                              dtype=np.dtype(self.config.get('cache_dtype','float16')),shape=(n,width)))
                labels.append(np.zeros(n,dtype=np.int8))
            def consume(frame,x,y,done):
                for j in range(len(pack)):
                    if done[j]:continue
                    if frame>=len(labels[j]):raise RuntimeError('Cache capacity exceeded')
                    arrays[j][frame]=x[j];labels[j][frame]=y[j];sizes[j]=frame+1
            result,_,_=self.run([r for _,r,_ in pack],profile=profile,collect_callback=consume,seed=42+offset)
            if result['accuracy']<.98 or any(result[k]['required'] and result[k]['coverage']<.98 for k in ('drumroll','swell')):
                raise RuntimeError('Expert/evaluator gate failed: '+json.dumps(result))
            for j,(i,row,fingerprint) in enumerate(pack):
                arrays[j].flush();(directory/f'{i:04d}.partial').replace(directory/f'{i:04d}.npy')
                np.save(directory/f'{i:04d}-labels.npy',labels[j][:sizes[j]])
                np.save(directory/f'{i:04d}-initial.npy',self.initial_motor[j])
                atomic_json(directory/f'{i:04d}.json',{'fingerprint':fingerprint,'frames':sizes[j],
                            'row':row,'profile':profile,'simulation_seed':42+offset,'expert_batch_metrics':result})
                log(self.root,'cache_complete',map=i,profile=profile,frames=sizes[j])
            del arrays

    def evaluate(self,rows,policy,profile='clean',threshold=.65,disconnect=False):
        reports=[];games=[]
        for offset in range(0,len(rows),self.config['batch']):
            report,pack,_=self.run(rows[offset:offset+self.config['batch']],policy,profile,threshold,disconnect,seed=42+offset)
            reports.append(report);games.extend(pack)
        result=all_metrics(games);result.update(profile=profile,
                inference_batch_p95_ms=max(r['inference_batch_p95_ms'] for r in reports),
                inference_batch_p50_ms=float(np.median([r['inference_batch_p50_ms'] for r in reports])))
        return result

def score(report):
    n=max(1,report['total_notes']);value=report['accuracy']-.2*report['false_hits']/n-.1*report['full_alt_violations']/n
    for kind in ('drumroll','swell'):
        if report[kind]['required']:value+=.1*report[kind]['coverage']
    bias=report.get('mean_signed_error_ms')
    if bias is not None:value-=.005*min(1.,abs(bias)/20.)
    return value

def sampled_features(data,indices,initial=None,lag=1):
    if lag<1:raise ValueError('lag must be positive')
    current=np.asarray(data[indices],dtype=np.float32)
    before=np.asarray(data[np.maximum(indices-lag,0)],dtype=np.float32).copy()
    before[indices<lag]=0 if initial is None else initial
    return np.concatenate((current,np.clip(current-before,-2,2)),axis=1)

def save_checkpoint(path,policy,optimizer,config,epoch,best,stale,threshold):
    payload={'architecture':VERSION,'config':config,'epoch':epoch,'best_score':best,'stale':stale,
             'threshold':threshold,'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
             'numpy_rng':np.random.get_state(),'random_rng':random.getstate()}
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)

def fit(engine,cache_dirs,validation,out,epochs=20,initial=None,phase='A',threshold=.65,
        selection_fn=None):
    out.mkdir(parents=True,exist_ok=True);size=len(np.load(ROOT/'graph_arrays.npz')['output'])*4
    cls=MotorPolicy
    if engine.config.get('motor_interface')=='continuous_modes':
        from motor_modes import ModeMotorPolicy
        cls=ModeMotorPolicy
    policy=cls(size,hidden=engine.config['decoder_hidden'],
                       input_dropout=engine.config['decoder_dropout'],
                       scale_floor=engine.config.get('scale_floor',.02)).cuda()
    optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3 if phase=='A' else 2e-4,weight_decay=.001)
    class_weights=torch.tensor(engine.config['class_weights'],device='cuda',dtype=torch.float32)
    entries=[p for d in cache_dirs for p in sorted(d.glob('*.json'))]
    if not entries:raise ValueError('No completed cache')
    means=[]
    for p in entries:
        meta=json.loads(p.read_text());data=np.load(p.with_suffix('.npy'),mmap_mode='r')
        initial_path=p.parent/(p.stem+'-initial.npy')
        initial_vector=np.load(initial_path) if initial_path.exists() else None
        idx=np.linspace(0,meta['frames']-1,min(256,meta['frames']),dtype=int)
        means.append(sampled_features(data,idx,initial_vector,engine.config.get('motor_delta_lag_frames',1)))
    policy.calibrate(np.concatenate(means));del means
    start=0;best=-1e9;stale=0
    if initial:
        payload=torch.load(initial,map_location='cuda',weights_only=False)
        if payload['architecture']!=VERSION:raise ValueError('Legacy checkpoint cannot load into v2')
        recalibrate=payload['config'].get('observation_style','legacy')!=engine.config.get('observation_style','legacy') or \
            payload['config'].get('retina_x_warp')!=engine.config.get('retina_x_warp') or \
            payload['config'].get('retina_bounds')!=engine.config.get('retina_bounds') or \
            payload['config'].get('motor_delta_lag_frames',1)!=engine.config.get('motor_delta_lag_frames',1)
        if recalibrate:mean,scale=policy.mean.clone(),policy.scale.clone()
        policy.load_state_dict(payload['policy']);threshold=payload['threshold']
        if recalibrate:
            policy.mean.copy_(mean);policy.scale.copy_(scale)
            log(engine.root,'visual_warm_start_recalibrated',source=str(initial))
    if (out/'last.pt').exists():
        payload=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
        if payload['config']!=engine.config:raise ValueError('Checkpoint config mismatch')
        policy.load_state_dict(payload['policy']);optimizer.load_state_dict(payload['optimizer'])
        start=payload['epoch'];best=payload['best_score'];stale=payload['stale'];threshold=payload['threshold']
        torch.set_rng_state(payload['torch_rng'].cpu());np.random.set_state(payload['numpy_rng']);random.setstate(payload['random_rng'])
        torch.cuda.set_rng_state_all([state.cpu() for state in payload['cuda_rng']])
        log(engine.root,'resume',phase=phase,epoch=start)
    for epoch in range(start,epochs):
        losses=[];entries.sort();random.shuffle(entries);policy.train()
        for p in entries:
            meta=json.loads(p.read_text());data=np.load(p.with_suffix('.npy'),mmap_mode='r')
            labels=np.load(p.parent/(p.stem+'-labels.npy')).astype(np.int64)
            initial_path=p.parent/(p.stem+'-initial.npy')
            initial_vector=np.load(initial_path) if initial_path.exists() else None
            positives=np.flatnonzero(labels>0);none=np.flatnonzero(labels==0)
            if not len(positives):continue
            # Train all negative frames, including intervals between presses.
            selected=np.arange(len(labels))
            np.random.shuffle(selected)
            if engine.config.get('smoke'):selected=np.tile(selected,max(1,int(np.ceil(8192/len(selected)))))
            for offset in range(0,len(selected),256):
                idx=selected[offset:offset+256]
                x=torch.as_tensor(sampled_features(data,idx,initial_vector,
                              engine.config.get('motor_delta_lag_frames',1)),device='cuda')
                y=torch.as_tensor(labels[idx],device='cuda');loss=torch.nn.functional.cross_entropy(policy(x),y,weight=class_weights)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),5);optimizer.step();losses.append(float(loss))
            log(engine.root,'train_map',phase=phase,epoch=epoch+1,map=p.stem,loss=float(np.mean(losses)))
        if not losses:raise RuntimeError('No optimizer steps')
        policy.eval();report=engine.evaluate(validation,policy,threshold=threshold)
        current=score(report);stress=None;selection=None
        if selection_fn is not None:
            current,selection=selection_fn(policy,threshold,report)
        if phase=='C':
            stress=engine.evaluate(validation,policy,profile='combined',threshold=threshold)
            current=.5*current+.5*score(stress)
        improved=current>best+1e-5
        if improved:best=current;stale=0
        else:stale+=1
        save_checkpoint(out/'last.pt',policy,optimizer,engine.config,epoch+1,best,stale,threshold)
        if improved:save_checkpoint(out/'best.pt',policy,optimizer,engine.config,epoch+1,best,stale,threshold)
        record={'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':report,
                'selection':selection,'stress':stress,'best':improved}
        atomic_json(out/f'epoch-{epoch+1}.json',record);log(engine.root,'epoch_complete',phase=phase,**record)
        # No premature epoch-one termination; still stop clearly failed or
        # degrading runs rather than burning the whole epoch budget unattended.
        if epoch>=3 and (stale>=6 or (report['accuracy']<.1 and report['false_hits']>report['total_notes'])):
            log(engine.root,'early_stop',phase=phase,epoch=epoch+1,reason='plateau_or_poor_validation');break
    return out/'best.pt'

def load_policy(path):
    payload=torch.load(path,map_location='cuda',weights_only=False)
    cls=MotorPolicy
    if payload['policy']['net.4.weight'].shape[0]==7:
        from motor_modes import ModeMotorPolicy
        cls=ModeMotorPolicy
    policy=cls(len(payload['policy']['mean']),hidden=payload['policy']['net.0.weight'].shape[0],
                       input_dropout=payload['config'].get('decoder_dropout',0.)).cuda()
    policy.load_state_dict(payload['policy']);policy.eval()
    return policy,payload

def gate(engine,rows,checkpoint):
    policy,payload=load_policy(checkpoint);threshold=payload['threshold']
    full=engine.evaluate(rows,policy,threshold=threshold)
    disconnected=engine.evaluate(rows,policy,threshold=threshold,disconnect=True)
    class BlankViewport(GameplayPixels):
        def frame(self,game,show_keys=False):return self.background()
    renderer=engine.renderer
    try:
        engine.renderer=BlankViewport()
        blank=engine.evaluate(rows,policy,threshold=threshold)
    finally:engine.renderer=renderer
    n=max(1,full['total_notes']);passed=(full['accuracy']>=.8 and full['false_hits']/n<=.2 and
           full['full_alt_violations']==0 and full['accuracy']-disconnected['accuracy']>=.2 and
           full['accuracy']-blank['accuracy']>=.2 and
           full['size_match_rate']>=.95 and full['big_note_full_hit_rate']>=.95 and
           (not full['drumroll']['required'] or full['drumroll']['coverage']>=.8) and
           (not full['swell']['required'] or full['swell']['coverage']>=.95))
    report={'passed':passed,'full':full,'disconnected':disconnected,'retinal_blank':blank,
            'visual_accuracy_gap':full['accuracy']-blank['accuracy'],'checkpoint':str(checkpoint)}
    return report

def publish_replay(engine,row,checkpoint,output):
    policy,payload=load_policy(checkpoint)
    metrics,games,trace=engine.run([row],policy,threshold=payload['threshold'],trace=True)
    keys=['NONE','F','J','D','K','F+J','D+K']
    atomic_json(output,{'schema_version':4,'metadata':row['metadata'],'beatmap':row,'metrics':metrics,
       'actions':[{**a,'action':keys[a['action_id']]} for a in games[0].actions],
       'neural_trace':trace,'provenance':engine.config,'checkpoint':str(checkpoint)})
    return metrics

def make_config(args):
    calibration=json.loads(Path('runs/malecns/calibration.json').read_text())
    if calibration['status']!='motor_signal_found':raise RuntimeError('Calibration gate failed')
    choice=calibration['selected']
    gain=getattr(args,'gain',None);tonic=getattr(args,'tonic',None)
    gain=choice['gain'] if gain is None else gain
    tonic=choice['tonic'] if tonic is None else tonic
    if not np.isfinite(gain) or gain<=0 or not np.isfinite(tonic) or tonic<0:
        raise ValueError('Gain must be positive and tonic must be nonnegative and finite')
    if not 1<=args.batch<=8:raise ValueError('Fixed CSR supports batch sizes 1..8')
    return {'architecture':VERSION,'seed':42,'batch':args.batch,'gain':gain,'tonic':tonic,
            'retina_bounds':[40/511,24/95,176/511,76/95],'retina_bilinear':True,
            'neuron_model':'graded-rate engineering hypothesis','rate_coupling':.95,
            'weight_transform':'signed measured contacts / incoming absolute sum * .95',
            'input_nodes':'photoreceptor activation externally driven by RGB; no input-node synaptic feedback',
            'motor_interface':'continuous_modes','training_label_policy':'circle onset <=12ms before target; 8ms consistent evidence before actuation; sustained native long-object modes',
            'frame_interval_ms':8.,'model_sampling_hz':125.,'presentation_fps_cap':120,
            'validation_objective':'accuracy/false-hit/long-coverage plus 0.005 timing-bias penalty per 20ms',
            'smoke':args.smoke,'retina':'RGB only; no teacher key lights','connectome':'MaleCNS v1.0, Traced, measured edges',
            'neurons':165122,'decoder':'measured motor activation + temporal difference; seven learned action/motor-mode categories',
            'interface':'probability hysteresis + deterministic per-color hand alternation',
            'synapse_backend':'fixed-tree CSR Triton, no atomic reduction',
            'motor_precision':'float32 live and cache; weak neural responses are retained','cache_dtype':'float32',
            'tonic_scope':'photo_targets','body_tonic':6.5,'photo_tonic':0.,'warmup_ms':512.,'scale_floor':1e-7,
            'legacy_lif_parameters_unused':['gain','tonic','tonic_scope','body_tonic','photo_tonic'],
            'decoder_hidden':128,'decoder_dropout':.1,'class_weights':[1,2,2,6,6,2,2],
            'pilot_songs':getattr(args,'pilot_songs',32),
            'phase_plan':'A only (disk-safe)' if getattr(args,'phase_a_only',False) else 'A and C',
            'cache_workers':getattr(args,'cache_workers',1),
            'graph_hashes':{k:v['sha256'] for k,v in json.loads((ROOT/'source_manifest.json').read_text())['files'].items()},
            'source_hashes':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in
                            ('malecns_training.py','malecns_system.py','deterministic_sparse.py',
                             'malecns_campaign.py','motor_policy.py','visual_taiko.py','taiko/parser.py',
                             'measured_rate_system.py','motor_modes.py','circle_region_labels.py',
                             'malecns_cache_parallel.py','malecns_cache_migration.py','refresh_native_split.py',
                             'malecns_relabel_cache.py')}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/malecns_v2'))
    p.add_argument('--dataset',type=Path,default=Path('native_songs/native_beatmaps.json'))
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--gain',type=float);p.add_argument('--tonic',type=float)
    p.add_argument('--pilot-songs',type=int,default=32)
    p.add_argument('--cache-workers',type=int,choices=(1,2,3),default=3)
    p.add_argument('--split-from',type=Path)
    p.add_argument('--target-source-sha256',help='Exact native chart for final replay; prevents alternate revisions')
    p.add_argument('--phase-a-only',action='store_true',help='Skip C cache/training when disk reserve cannot hold both phases')
    p.add_argument('--smoke',action='store_true');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();seed_all(42);torch.set_num_threads(2);root=args.root;root.mkdir(parents=True,exist_ok=True)
    config=make_config(args)
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text())!=config:
        raise ValueError('Existing run source/config changed; use a new root, preserve old artifacts')
    atomic_json(root/'config.json',config)
    if (root/'split.json').exists():split=json.loads((root/'split.json').read_text())
    elif args.split_from:
        split=json.loads(args.split_from.read_text());atomic_json(root/'split.json',split)
    else:
        rows=json.loads(args.dataset.read_text());split,excluded=choose_split(rows);del rows
        atomic_json(root/'split.json',split);atomic_json(root/'excluded_maps.json',excluded)
    if args.prepare_only:
        log(root,'prepared',counts={k:len(v) for k,v in split.items()});return
    if not (root/'expert_preflight.json').exists():audit_expert_split(split,root,fast=config.get('motor_interface')=='continuous_modes')
    elif not all(r['passed'] for r in json.loads((root/'expert_preflight.json').read_text())):
        raise RuntimeError('Previous expert preflight failed; refusing GPU training')
    engine=Engine(config,root)
    if args.smoke:
        from malecns_campaign import smoke_rows
        examples=smoke_rows(split['A'])
        engine.cache(examples,root/'cache_A')
        checkpoint=fit(engine,[root/'cache_A'],examples,root/'phase_A',args.epochs)
        result=gate(engine,examples,checkpoint);atomic_json(root/'A_gate.json',result)
        for i,row in enumerate(examples):publish_replay(engine,row,checkpoint,root/f'replay-{i}.json')
        atomic_json(root/'result.json',{'status':'passed' if result['passed'] else 'gate_failed','A_gate':result})
        log(root,'completed',status='passed' if result['passed'] else 'gate_failed');return
    # A short native pilot is a structural/resource guard before multi-hour cache.
    # Validation and test songs remain disjoint; test is never used for selection.
    if not 1<=args.pilot_songs<=len(split['A']):raise ValueError('Pilot must stay within selected A songs')
    train_probes=probes(split['A'][:args.pilot_songs],length=8000);validation=probes(split['validation'],length=8000)
    pilot_rows=pilot_validation(split['validation'])
    engine.cache(train_probes,root/'cache_pilot')
    if (root/'migration.json').exists() and (root/'pilot/best.pt').exists():
        pilot=root/'pilot/best.pt'
        log(root,'imported_pilot_revalidation',checkpoint=str(pilot))
    else:pilot=fit(engine,[root/'cache_pilot'],pilot_rows,root/'pilot',min(args.epochs,12))
    pilot_gate=gate(engine,pilot_rows,pilot);atomic_json(root/'pilot_gate.json',pilot_gate)
    if not pilot_gate['passed']:
        log(root,'gate_failed',stage='held_out_pilot',metrics=pilot_gate);return
    from malecns_cache_parallel import cache_parallel
    cache_parallel(engine,split['A'],root/'cache_A',workers=args.cache_workers,stage='A')
    best=fit(engine,[root/'cache_A'],validation,root/'phase_A',args.epochs,initial=pilot)
    a_gate=gate(engine,validation,best);atomic_json(root/'A_gate.json',a_gate)
    if not a_gate['passed']:log(root,'gate_failed',stage='A',metrics=a_gate);return
    accepted=False;selected=best
    if args.phase_a_only:
        log(root,'phase_C_skipped',reason='Disk reserve; v16 C checkpoint was not selected')
    else:
        a_policy,a_payload=load_policy(best)
        before={k:engine.evaluate(validation,a_policy,k,threshold=a_payload['threshold']) for k in ('latency','jitter','combined')}
        atomic_json(root/'stress_before.json',before)
        cache_parallel(engine,split['C'],root/'cache_C',profile='combined',workers=args.cache_workers,stage='C')
        candidate=fit(engine,[root/'cache_A',root/'cache_C'],validation,root/'phase_C',min(args.epochs,12),initial=best,phase='C')
        c_policy,c_payload=load_policy(candidate)
        clean=engine.evaluate(validation,c_policy,threshold=c_payload['threshold'])
        after={k:engine.evaluate(validation,c_policy,k,threshold=c_payload['threshold']) for k in ('latency','jitter','combined')}
        accepted=clean['accuracy']>=a_gate['full']['accuracy']-.005 and sum(score(v) for v in after.values())>sum(score(v) for v in before.values())
        atomic_json(root/'C_decision.json',{'accepted':accepted,'clean':clean,'before':before,'after':after})
        selected=candidate if accepted else best
    policy,payload=load_policy(selected)
    final_val=engine.evaluate(split['validation'],policy,threshold=payload['threshold'])
    unseen=engine.evaluate(split['test'],policy,threshold=payload['threshold'])
    atomic_json(root/'full_validation.json',final_val);atomic_json(root/'unseen_test.json',unseen)
    # Replay the user's requested IDEALESS target with every native long object.
    sources=json.loads(args.dataset.read_text())
    targets=[r for r in sources if r['source_sha256']==args.target_source_sha256] if args.target_source_sha256 else [
        r for r in sources if r['metadata']['title'].strip().upper()=='IDEALESS IDEOLOGY']
    if args.target_source_sha256 and len(targets)!=1:raise ValueError('Explicit target native source hash is absent/ambiguous')
    target=targets[0] if targets else split['test'][0]
    # The imported dataset records the previous renderer's per-object speed.
    # Reparse the hash-verified native .osu before publishing a v18 replay.
    from refresh_native_split import refresh
    target=refresh(target)
    output=Path('web-fly/public/demos/malecns-v18-final-replay.json')
    replay=publish_replay(engine,target,selected,output)
    atomic_json(root/'result.json',{'status':'completed','selected_checkpoint':str(selected),
                'full_validation':final_val,'unseen':unseen,'replay':replay,
                'C_accepted':accepted,'C_skipped':args.phase_a_only})
    log(root,'completed',checkpoint=str(selected),replay=str(output))

if __name__=='__main__':main()
