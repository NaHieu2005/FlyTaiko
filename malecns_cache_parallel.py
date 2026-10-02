"""Three isolated GPU cache processes, disjoint indices and atomic completion.

Long/short maps are duration-bucketed to reduce idle columns. Training/evaluation
remain single-process. Completed caches survive interruption; no shared writers.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import torch
from neural_campaign import atomic_json
from neural_system import seed_all

def assignments(rows,missing,batch,workers):
    from malecns_training import endpoint
    if len(missing)!=len(set(missing)):raise ValueError('Duplicate cache indices')
    if not 1<=workers<=3 or not 1<=batch<=8:raise ValueError('Unsafe worker/batch count')
    ordered=sorted(missing,key=lambda i:(-endpoint(rows[i]),i));jobs=[[] for _ in range(workers)];loads=[0.]*workers
    for start in range(0,len(ordered),batch):
        pack=ordered[start:start+batch];worker=min(range(workers),key=lambda w:loads[w])
        jobs[worker].extend(pack);loads[worker]+=max(endpoint(rows[i]) for i in pack)
    return jobs

def missing_indices(rows,directory,config,profile):
    from malecns_cache_migration import fingerprint
    result=[]
    for i,row in enumerate(rows):
        path=directory/f'{i:04d}.json'
        if not path.exists():result.append(i)
        elif json.loads(path.read_text())['fingerprint']!=fingerprint(row,config,profile):
            raise ValueError('Stale cache: '+str(path))
    return result

def cache_parallel(engine,rows,directory,profile='clean',workers=3,stage='A'):
    from malecns_campaign import log
    directory.mkdir(parents=True,exist_ok=True)
    missing=missing_indices(rows,directory,engine.config,profile)
    if not missing:
        log(engine.root,'cache_stage_complete',stage=stage,maps=len(rows),reused=len(rows));return
    if workers==1:engine.cache(rows,directory,profile);return
    # Free the parent's whole graph before loading three independent graphs.
    engine.brain=None;gc.collect();torch.cuda.empty_cache()
    work=directory/'.workers';work.mkdir(exist_ok=True)
    rows_path=work/'rows.json';atomic_json(rows_path,rows)
    jobs=assignments(rows,missing,engine.config['batch'],workers)
    atomic_json(work/'plan.json',{'stage':stage,'profile':profile,'jobs':jobs,'batch':engine.config['batch']})
    log(engine.root,'parallel_cache_start',stage=stage,workers=sum(bool(j) for j in jobs),
        reused=len(rows)-len(missing),missing=len(missing),maps_per_worker=[len(j) for j in jobs])
    processes=[];handles=[];readers=[];positions=[]
    previous_handler=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise KeyboardInterrupt('Cache coordinator terminated')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for worker,indices in enumerate(jobs):
            if not indices:continue
            index_path=work/f'worker-{worker}-indices.json';atomic_json(index_path,indices)
            worker_root=work/f'worker-{worker}';worker_root.mkdir(exist_ok=True)
            stdout_path=work/f'worker-{worker}.log'
            handle=stdout_path.open('a');handles.append(handle)
            reader=stdout_path.open();reader.seek(0,2);readers.append(reader);positions.append(worker)
            env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                 'OPENBLAS_NUM_THREADS':'1','PYTHONFAULTHANDLER':'1'}
            command=[sys.executable,'-u','malecns_cache_parallel.py','--worker',str(worker),
                '--root',str(engine.root),'--rows',str(rows_path),'--indices',str(index_path),
                '--directory',str(directory),'--profile',profile,'--worker-root',str(worker_root)]
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env)
            processes.append(process);log(engine.root,'cache_worker_started',stage=stage,worker=worker,pid=process.pid)
        last_progress=0.
        while True:
            for worker,reader in zip(positions,readers):
                for line in reader.readlines():
                    try:record=json.loads(line);record.update(worker=worker,cache_stage=stage);print(json.dumps(record),flush=True)
                    except json.JSONDecodeError:print(f'[cache worker {worker}] '+line.rstrip(),flush=True)
            failed=[(w,p.returncode) for w,p in zip(positions,processes) if p.poll() not in (None,0)]
            if failed:raise RuntimeError('Cache worker failed, preserving completed maps: '+str(failed))
            if time.monotonic()-last_progress>30:
                memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith(('MemAvailable:','MemTotal:'))}
                available=memory['MemAvailable']/1024**2
                if available<3:raise RuntimeError('Host RAM safety reserve <3GiB; stopping cache workers safely')
                done=len(rows)-len(missing_indices(rows,directory,engine.config,profile))
                log(engine.root,'cache_progress',stage=stage,completed=done,total=len(rows),
                    active_workers=sum(p.poll() is None for p in processes),ram_available_gib=available);last_progress=time.monotonic()
            if all(p.poll() is not None for p in processes):break
            time.sleep(2)
        remaining=missing_indices(rows,directory,engine.config,profile)
        if remaining:raise RuntimeError('Incomplete parallel cache: '+str(remaining))
        log(engine.root,'cache_stage_complete',stage=stage,maps=len(rows),workers=workers)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for handle in handles+readers:handle.close()
        signal.signal(signal.SIGTERM,previous_handler)

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True)
    for name in ('root','rows','indices','directory','worker-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',choices=('clean','combined'),required=True);a=p.parse_args()
    seed_all(42);torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.24)
    from malecns_training import Engine
    config=json.loads((a.root/'config.json').read_text());rows=json.loads(a.rows.read_text());indices=json.loads(a.indices.read_text())
    # Honor duration order while retaining stable global cache filenames.
    engine=Engine(config,a.worker_root)
    # Keep each transaction small even if the simulation batch supports eight.
    # A native CUDA crash then loses at most four maps, not an entire long pack.
    for start in range(0,len(indices),4):
        engine.cache(rows,a.directory,a.profile,indices=indices[start:start+4])

if __name__=='__main__':main()
