"""Motor-neuron-only readout and explicit deterministic neuron-to-key interface."""
import numpy as np
import torch
from torch import nn

CATEGORIES=('none','don','kat','don_big','kat_big')
ACTION_TO_CATEGORY=np.array([0,1,1,2,2,3,4],dtype=np.int64)

class MotorPolicy(nn.Module):
    def __init__(self,size,hidden=256,input_dropout=0.,scale_floor=.02):
        super().__init__()
        self.register_buffer('mean',torch.zeros(size))
        self.register_buffer('scale',torch.ones(size))
        self.input_dropout=input_dropout
        self.scale_floor=scale_floor
        self.net=nn.Sequential(nn.Linear(size,hidden),nn.SiLU(),nn.Linear(hidden,128),nn.SiLU(),nn.Linear(128,5))

    @torch.no_grad()
    def calibrate(self,features):
        features=torch.as_tensor(features,dtype=torch.float32,device=self.mean.device)
        self.mean.copy_(features.mean(0));self.scale.copy_(features.std(0).clamp_min(self.scale_floor))

    def forward(self,features):
        # All channels are measured output-neuron voltage/rate features; no pixels,
        # beatmap state, clock or action history reach this network.
        normalized=((features-self.mean)/self.scale).clamp(-8,8)
        normalized=nn.functional.dropout(normalized,p=self.input_dropout,training=self.training)
        return self.net(normalized)

class KeyInterface:
    def __init__(self,threshold=.65,release=.35):
        if not 0<=release<threshold<=1:raise ValueError('Invalid trigger hysteresis')
        self.threshold,self.release=threshold,release;self.latched=False
        self.right={'don':False,'kat':False}

    def decide(self,probabilities,delta_ms=8.):
        p=np.asarray(probabilities);hit=1-float(p[0])
        if hit<self.release:self.latched=False
        if self.latched or hit<self.threshold:return 0
        category=int(np.argmax(p[1:]))+1;self.latched=True
        if category in (3,4):return category+2
        color='don' if category==1 else 'kat'
        action=(1 if color=='don' else 3)+int(self.right[color])
        self.right[color]=not self.right[color]
        return action

def output_features(brain):
    voltage=((brain.v[brain.output]+52.)/7.).clamp(-2,2)
    rate=(brain.rates[brain.output]/100.).clamp(0,5)
    return torch.cat((voltage,rate),dim=0).T.contiguous()
