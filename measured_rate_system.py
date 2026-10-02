"""Experimental continuous-activity dynamics on the measured MaleCNS graph.

This is a rate/graded modelling hypothesis, NOT a validated biophysical brain.
Topology, body IDs, transmitter signs and relative contact counts are retained.
Incoming absolute weights are normalized for a contractive recurrent update;
this transformation is explicit and never replaces the original contact data.
No RGB-to-output shortcut exists. RGB drives measured photoreceptors only.
"""
import numpy as np
from scipy import sparse
import torch
from malecns_system import MeasuredBrain
from deterministic_sparse import FixedCSR
from prepare_malecns import ROOT

class MeasuredRateBrain(MeasuredBrain):
    def __init__(self,root=ROOT,batch=1,device='cuda',coupling=.95,baseline=.5,**kwargs):
        if not 0<=coupling<1:raise ValueError('Coupling must be contractive')
        super().__init__(root,batch=batch,device=device,gain=1.,tonic=0.)
        contacts=sparse.load_npz(str(root)+'/contacts_pre_post.npz')
        weights=sparse.diags(self.arrays['signs']).dot(contacts).T.tocsr()
        total=np.asarray(abs(weights).sum(axis=1)).ravel()
        weights=sparse.diags(coupling/np.maximum(total,1)).dot(weights).tocsr()
        weights.eliminate_zeros()
        self.W=torch.sparse_csr_tensor(torch.as_tensor(weights.indptr,device=device),
            torch.as_tensor(weights.indices,device=device),
            torch.as_tensor(weights.data,dtype=torch.float32,device=device),size=weights.shape)
        self.fixed_synapses=FixedCSR(weights,device) if self.device.type=='cuda' else None
        self.baseline=baseline;self.coupling=coupling
        self.model='measured signed contact-normalized graded-rate hypothesis'

    @torch.no_grad()
    def advance(self,images,duration=8.,disconnect=False):
        steps=round(duration/self.dt)
        if abs(steps*self.dt-duration)>1e-5:raise ValueError('Duration must align to neuron timestep')
        retinal=self.retina.sample(images)
        if self.elapsed==0:self.graded_state=torch.zeros_like(self.v)
        for _ in range(steps):
            transmitted=self.delay[self.cursor]
            syn=0. if disconnect else (self.fixed_synapses(transmitted) if self.fixed_synapses is not None
                                       else torch.sparse.mm(self.W,transmitted))
            target=(torch.full_like(self.v,self.baseline) if isinstance(syn,float)
                    else torch.clamp(self.baseline+syn,min=0))
            target[self.retina.photo]=retinal
            state=self.graded_state
            state=state+(self.dt/5.)*(target-state)
            self.v=-52.+7.*state
            self.graded_state=state
            # Rate readout is an engineering population-activity proxy. There
            # are no discrete spikes in this model; never fabricate spike traces.
            self.rates=100.*state;self.spikes=torch.zeros_like(state)
            self.delay[self.cursor]=state;self.cursor=(self.cursor+1)%self.delay_steps
        self.elapsed+=duration
        return torch.cat(((self.v[self.output]+52.)/7.,self.rates[self.output]/100.),dim=0).T.contiguous()

    def trace(self,time_ms,index=0):
        record=super().trace(time_ms,index)
        record['neuron_model']=self.model
        record['activity_units']='100*continuous activation; NOT measured biological Hz or discrete spikes'
        return record
