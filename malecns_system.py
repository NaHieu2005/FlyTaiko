"""Measured MaleCNS graph, image-only retina and delayed spiking dynamics.

Topology/contact counts are measured. Transmitter signs, neuron dynamics,
image projection and the key readout are explicit modelling assumptions.
No synthetic edges or fallback connectome are permitted.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow as pa
from scipy import sparse
import torch

from prepare_malecns import ROOT, FILES

def build_graph(root=ROOT):
    root=Path(root)
    annotations=feather.read_table(root/FILES[0]).to_pandas()
    neurons=annotations[annotations.status=='Traced'].sort_values('bodyId').reset_index(drop=True)
    ids=neurons.bodyId.to_numpy(dtype=np.int64)
    nt=feather.read_table(root/FILES[1],columns=['body','consensus_nt']).to_pandas()
    transmitter=nt.set_index('body').consensus_nt.reindex(ids).fillna('unclear').to_numpy()
    # A simplified presynaptic sign rule, not a receptor-specific measurement.
    signs=np.array([1 if n=='acetylcholine' else -1 if n in ('gaba','glutamate','histamine') else 0
                    for n in transmitter],dtype=np.float32)
    source=pa.memory_map(str(root/FILES[2]),'r')
    edges=pa.ipc.open_file(source)
    columns=edges.schema.names
    print({'edge_schema':columns},flush=True)
    pre_name=next(n for n in ('body_pre','bodyId_pre','pre') if n in columns)
    post_name=next(n for n in ('body_post','bodyId_post','post') if n in columns)
    weight_name=next(n for n in ('weight','syn_count','count') if n in columns)
    index=pd.Index(ids)
    pres=[];posts=[];counts=[];raw_rows=0;removed=0
    for b in range(edges.num_record_batches):
        chunk=edges.get_batch(b)
        pre=index.get_indexer(chunk.column(columns.index(pre_name)).to_numpy())
        post=index.get_indexer(chunk.column(columns.index(post_name)).to_numpy())
        retained=(pre>=0)&(post>=0);raw_rows+=len(pre);removed+=int((~retained).sum())
        pres.append(pre[retained].astype(np.int32));posts.append(post[retained].astype(np.int32))
        counts.append(chunk.column(columns.index(weight_name)).to_numpy()[retained].astype(np.float32))
    pre=np.concatenate(pres);post=np.concatenate(posts);contacts=np.concatenate(counts)
    adjacency=sparse.coo_matrix((contacts,(pre,post)),shape=(len(ids),len(ids))).tocsr()
    # For unlocated photoreceptors, infer retinal columns from their strongest
    # measured contact to a neuron with an annotated optic-lobe hex column.
    # This is an inferred screen projection, not a measured viewing direction.
    photo=np.flatnonzero(neurons.type.fillna('').str.match(r'^R(?:1-R6|[78].*)$').to_numpy())
    hexes=neurons[['assignedOlHex1','assignedOlHex2']].to_numpy(dtype=np.float32)
    locations=hexes[photo].copy(); inferred=0
    for j,i in enumerate(photo):
        if np.isfinite(locations[j]).all():continue
        s,e=adjacency.indptr[i:i+2];targets=adjacency.indices[s:e]
        valid=np.isfinite(hexes[targets]).all(1)
        if valid.any():
            winner=np.argmax(np.where(valid,adjacency.data[s:e],-1))
            locations[j]=hexes[targets[winner]];inferred+=1
    located=np.isfinite(locations).all(1);photo=photo[located];locations=locations[located]
    if not len(photo):raise ValueError('No anatomically grounded retinal column assignment')
    lower=np.min(locations,axis=0);upper=np.max(locations,axis=0)
    uv=(locations-lower)/np.maximum(upper-lower,1)
    output=np.flatnonzero(neurons.superclass.isin(['descending_neuron','vnc_motor','cb_motor']).to_numpy())
    if not len(output):raise ValueError('No measured motor/descending readout')
    if np.intersect1d(photo,output).size:raise ValueError('Input/readout overlap')
    # Trace positions retain missing anatomy as null; never fabricate positions.
    metadata=[{'body_id':str(row.bodyId),'type':row.type if isinstance(row.type,str) else '',
               'group':row.superclass if isinstance(row.superclass,str) else '',
               'side':row.rootSide if isinstance(row.rootSide,str) else '',
               'soma':None if row.somaLocation is None else [int(v) for v in row.somaLocation]}
              for row in neurons.itertuples()]
    sparse.save_npz(root/'contacts_pre_post.npz',adjacency)
    np.savez(root/'graph_arrays.npz',ids=ids,signs=signs,photo=photo,uv=uv,output=output,
             transmitters=transmitter.astype(str),photo_types=neurons.type.iloc[photo].to_numpy(dtype=str))
    (root/'neurons.json').write_text(json.dumps(metadata))
    report={'dataset':'MaleCNS v1.0','filter':'status == Traced','neurons':len(ids),
            'retained_edges':int(adjacency.nnz),'raw_rows':raw_rows,'removed_rows':removed,
            'photoreceptors':len(photo),'inferred_retinal_columns':inferred,
            'output_neurons':len(output),'sign_counts':{str(s):int((signs==s).sum()) for s in (-1,0,1)},
            'synthetic_edges':0,'adjacency_layout':'pre,post',
            'sign_assumption':'ACh +; GABA/glutamate/histamine -; unclear/monoamines 0',
            'vision_assumption':'Annotated/inferred hex columns linearly projected to RGB gameplay image'}
    (root/'graph_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return report

class ImageRetina:
    def __init__(self,arrays,device):
        self.photo=torch.as_tensor(arrays['photo'],device=device)
        self.uv=arrays['uv'];self.types=arrays['photo_types']
        self.bounds=(0.,0.,1.,1.);self.bilinear=False
        self.x_warp=None

    def sample(self,images):
        """Only accepts BxHxWx3 RGB pixels; no game state or note labels."""
        images=np.asarray(images,dtype=np.float32)
        if images.ndim!=4 or images.shape[-1]!=3:raise ValueError('Expected BxHxWx3 RGB')
        h,w=images.shape[1:3]
        left,top,right,bottom=self.bounds
        if not (0<=left<right<=1 and 0<=top<bottom<=1):raise ValueError('Invalid retinal camera bounds')
        u=self.uv[:,0]
        if self.x_warp is None:
            xf=(left+u*(right-left))*(w-1)
        else:
            split,focus=self.x_warp
            if not (0<split<1 and left<focus<right):
                raise ValueError('Invalid foveated retina x warp')
            xf=np.where(u<=split,left+(u/split)*(focus-left),
                        focus+((u-split)/(1-split))*(right-focus))*(w-1)
        yf=(top+self.uv[:,1]*(bottom-top))*(h-1)
        if self.bilinear:
            x=np.floor(xf).astype(int);y=np.floor(yf).astype(int)
            dx=(xf-x)[None,:,None];dy=(yf-y)[None,:,None]
            x1=np.minimum(x+1,w-1);y1=np.minimum(y+1,h-1)
            pixels=((1-dy)*((1-dx)*images[:,y,x]+dx*images[:,y,x1])+
                    dy*((1-dx)*images[:,y1,x]+dx*images[:,y1,x1])).astype(np.float32)/255.
        else:
            x=np.rint(xf).astype(int);y=np.rint(yf).astype(int)
            pixels=images[:,y,x,:]/255.
        luminance=pixels@np.array([.2126,.7152,.0722],dtype=np.float32)
        # RGB proxies only: a game framebuffer contains no ultraviolet channel.
        for kind,channel in (('R8p',2),('R8y',1)):
            mask=self.types==kind
            luminance[:,mask]=pixels[:,mask,channel]
        return torch.as_tensor(luminance.T,device=self.photo.device)

class MeasuredBrain:
    def __init__(self,root=ROOT,batch=1,device='cuda',gain=.275,tonic=0.,dt=.5,
                 tonic_scope='all',body_tonic=0.,photo_tonic=None):
        root=Path(root);self.arrays=np.load(root/'graph_arrays.npz',allow_pickle=False)
        self.ids=self.arrays['ids'];self.batch=batch;self.device=torch.device(device)
        self.dt,self.gain,self.tonic=dt,gain,tonic
        contacts=sparse.load_npz(root/'contacts_pre_post.npz')
        signed=sparse.diags(self.arrays['signs']).dot(contacts).T.tocsr()
        signed.eliminate_zeros()
        self.W=torch.sparse_csr_tensor(torch.as_tensor(signed.indptr,device=device),
                  torch.as_tensor(signed.indices,device=device),
                  torch.as_tensor(signed.data*gain,dtype=torch.float32,device=device),size=signed.shape)
        self.fixed_synapses=None
        if self.device.type=='cuda':
            from deterministic_sparse import FixedCSR
            signed.data=(signed.data*gain).astype(np.float32)
            self.fixed_synapses=FixedCSR(signed,self.device)
        self.retina=ImageRetina(self.arrays,device)
        self.tonic_scope=tonic_scope
        if tonic_scope=='photo_targets':
            targets=np.unique(contacts[self.arrays['photo']].indices)
            self.tonic=torch.full((len(self.ids),1),body_tonic,device=device)
            self.tonic[torch.as_tensor(targets,device=device)]=tonic
            if photo_tonic is not None:self.tonic[self.retina.photo]=photo_tonic
        elif tonic_scope!='all':raise ValueError('Unknown tonic scope')
        self.output=torch.as_tensor(self.arrays['output'],device=device)
        self.metadata=json.loads((root/'neurons.json').read_text())
        self.groups={name:np.array([i for i,n in enumerate(self.metadata) if n['group']==name])
                     for name in sorted({n['group'] for n in self.metadata})}
        self.sample=np.concatenate([v[::max(1,len(v)//8)][:8] for v in self.groups.values()]).tolist()
        self.v=torch.full((len(self.ids),batch),-52.,device=device)
        self.g=torch.zeros_like(self.v);self.refractory=torch.zeros_like(self.v)
        self.spikes=torch.zeros_like(self.v);self.rates=torch.zeros_like(self.v)
        # Delay rounded explicitly to integration grid (1.8 -> 2ms at .5ms).
        self.delay_steps=max(1,round(1.8/dt));self.delay=[torch.zeros_like(self.v) for _ in range(self.delay_steps)]
        self.cursor=0;self.elapsed=0.

    @torch.no_grad()
    def advance(self,images,duration=8.,disconnect=False):
        steps=round(duration/self.dt)
        if abs(steps*self.dt-duration)>1e-5:raise ValueError('Duration must align to neuron timestep')
        drive=self.retina.sample(images)*35.
        for _ in range(steps):
            transmitted=self.delay[self.cursor]
            syn=0. if disconnect else (self.fixed_synapses(transmitted) if self.fixed_synapses is not None
                                      else torch.sparse.mm(self.W,transmitted))
            self.g=self.g*np.exp(-self.dt/5.)+syn
            candidate=self.v+(self.dt/20.)*(-52.-self.v+self.g+self.tonic)
            candidate[self.retina.photo]+=(self.dt/20.)*drive
            candidate=torch.where(self.refractory>0,-52.,candidate)
            spikes=(candidate>-45.).float()
            self.v=torch.where(spikes.bool(),-52.,candidate)
            self.refractory=torch.where(spikes.bool(),2.2,torch.clamp(self.refractory-self.dt,min=0))
            self.rates=self.rates*np.exp(-self.dt/20.)+spikes*(1-np.exp(-self.dt/20.))*(1000/self.dt)
            self.spikes=spikes;self.delay[self.cursor]=spikes;self.cursor=(self.cursor+1)%self.delay_steps
        self.elapsed+=duration
        return torch.cat(((self.v[self.output]+52.)/7.,self.rates[self.output]/100.),dim=0).T.contiguous()

    def trace(self,time_ms,index=0):
        # Deterministic, group-stratified measured body IDs.
        sample=self.sample;groups={}
        for name,members in self.groups.items():
            if len(members)==0:continue
            ids=torch.tensor(members,device=self.device)
            groups[name]={'rate_hz':float(self.rates[ids,index].mean()),
                          'voltage_mv':float(self.v[ids,index].mean()),'spikes':int(self.spikes[ids,index].sum())}
        return {'time_ms':time_ms,'body_ids':[str(self.ids[i]) for i in sample],
                'neurons':[self.metadata[i] for i in sample],
                'voltage_mv':self.v[sample,index].cpu().tolist(),
                'rates_hz':self.rates[sample,index].cpu().tolist(),'groups':groups}

if __name__=='__main__':build_graph()
