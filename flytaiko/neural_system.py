"""Versioned, neural-only Taiko policy and batched causal LIF simulator.

Synthetic connectome adjacency is explicitly [pre, post]. All synaptic
currents use its transpose. Legacy pipeline/checkpoints are left untouched.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from scipy import sparse

from flyconnectome.loader import generate_synthetic_connectome
from flytaiko.flytaiko_pipeline import Decoder, JsonBeatmap, seed_all
from taiko.environment import ACTION_NAMES, NUM_ACTIONS


@dataclass
class NeuralConfig:
    version: str = 'neural-v5'
    seed: int = 42
    neurons: int = 10000
    step_ms: int = 8
    lif_dt_ms: float = .5
    tau_ms: float = 8.
    synaptic_gain: float = 300.
    sensory_gain: float = 90.
    tonic_current: float = 8.
    visible_ms: float = 1200.
    lookahead: int = 4
    hidden: int = 256
    sensory_relay: bool = True
    edge_triggered: bool = True

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class NeuralLIF:
    def __init__(self, adjacency, cfg, batch=1, device='cuda'):
        self.cfg = cfg
        self.device = torch.device(device)
        self.N, self.batch = adjacency.shape[0], batch
        # Normalize absolute incoming strength, preserve signs and topology.
        incoming = adjacency.T.tocsr().astype(np.float32)
        sums = np.asarray(abs(incoming).sum(axis=1)).ravel()
        incoming = sparse.diags(cfg.synaptic_gain / np.maximum(sums, 1.)).dot(incoming).tocoo()
        self.W = torch.sparse_coo_tensor(
            torch.as_tensor(np.vstack((incoming.row, incoming.col)), device=device),
            torch.as_tensor(incoming.data, device=device), (self.N, self.N)).coalesce()
        self.voltage = torch.full((self.N, batch), -65., device=device)
        self.spikes = torch.zeros_like(self.voltage)
        self.refractory = torch.zeros_like(self.voltage)
        self.rates = torch.zeros_like(self.voltage)
        self.spike_totals = torch.zeros(batch, device=device)
        self.steps = 0

    def reset(self):
        self.voltage.fill_(-65.); self.spikes.zero_(); self.refractory.zero_()
        self.rates.zero_(); self.spike_totals.zero_(); self.steps = 0

    @torch.no_grad()
    def step(self, current, dt_ms=None):
        dt = self.cfg.lif_dt_ms if dt_ms is None else dt_ms
        synaptic = torch.sparse.mm(self.W, self.spikes)
        candidate = self.voltage + (dt / self.cfg.tau_ms) * (-(self.voltage + 65.) + synaptic + current)
        candidate = torch.where(self.refractory > 0, -65., candidate)
        spikes = (candidate >= -50.).float()
        self.voltage = torch.where(spikes.bool(), -65., candidate)
        self.refractory = torch.where(spikes.bool(), 2., torch.clamp(self.refractory - dt, min=0))
        self.spikes = spikes
        alpha = 1 - np.exp(-dt / 8.)
        self.rates = self.rates * (1 - alpha) + spikes * alpha
        self.spike_totals += spikes.sum(0)
        self.steps += 1
        return spikes

    @torch.no_grad()
    def advance(self, current, duration_ms):
        full = int(duration_ms / self.cfg.lif_dt_ms)
        for _ in range(full):
            self.step(current)
        remainder = duration_ms - full * self.cfg.lif_dt_ms
        if remainder > 1e-6:
            self.step(current, remainder)

    def readout(self, ids):
        voltage = ((self.voltage[ids] + 65.) / 15.).clamp(-1., 2.)
        return torch.cat((voltage, self.rates[ids]), dim=0).T.contiguous()


class NeuralDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden=256):
        super().__init__()
        self.normalize = torch.nn.LayerNorm(input_size, elementwise_affine=False)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, NUM_ACTIONS))

    def forward(self, features):
        return self.net(self.normalize(features))


class NeuralRuntime:
    def __init__(self, cfg, batch=1, device='cuda'):
        self.cfg, self.device, self.batch = cfg, torch.device(device), batch
        self.conn = generate_synthetic_connectome(cfg.neurons, seed=cfg.seed)
        if cfg.sensory_relay:
            # Explicit synthetic visual relay: sparse, directed sensory ->
            # visual-projection contacts. This is a documented model circuit,
            # not a claim about measured fly synapses. Inputs still cross a
            # spiking synapse; no raw feature reaches the decoder.
            post = np.array([n.idx for n in self.conn.neurons if n.cell_type=='visual_proj'])
            pre = self.conn.sensory_ids[(np.arange(len(post))*len(self.conn.sensory_ids)//len(post)).astype(int)]
            relay=sparse.coo_matrix((np.full(len(post),100.,dtype=np.float32),(pre,post)),
                                   shape=self.conn.adjacency.shape).tocsr()
            self.conn.adjacency=(self.conn.adjacency+relay).tocsr()
        self.lif = NeuralLIF(self.conn.adjacency, cfg, batch, device)
        self.sensory = torch.as_tensor(self.conn.sensory_ids, device=device)
        self.readout = torch.as_tensor(self.conn.readout_ids, device=device)
        self.decoder = NeuralDecoder(len(self.readout) * 2, cfg.hidden).to(device)
        self.current = torch.zeros((cfg.neurons, batch), device=device)

    def observe(self, env):
        state = env.get_state(self.cfg.lookahead)
        state['upcoming_notes'] = [(dt, kind) if -env.windows['miss'] <= dt <= self.cfg.visible_ms
                                   else (99999, 'none') for dt, kind in state['upcoming_notes']]
        return state

    def encode(self, states):
        values = np.zeros((len(self.sensory), self.batch), dtype=np.float32)
        width = len(self.sensory) // (self.cfg.lookahead + 2)
        q = width // 6
        for b, state in enumerate(states):
            for i, (dt, kind) in enumerate(state['upcoming_notes'][:self.cfg.lookahead]):
                if kind == 'none' or dt > self.cfg.visible_ms:
                    continue
                urgency = 1 / (1 + max(dt, 0) / 150)
                peak = max(0, 1 - abs(dt) / 24)
                start = i * width
                channels = [urgency, peak, float('don' in kind) * urgency,
                            float('kat' in kind) * urgency, float('big' in kind) * urgency,
                            min(max(dt, 0) / self.cfg.visible_ms, 1)]
                for j, value in enumerate(channels):
                    values[start + j*q:start + (j+1)*q, b] = value * self.cfg.sensory_gain
            for j, key in enumerate(('don_alt', 'kat_alt')):
                start = (self.cfg.lookahead + j) * width
                half = width // 2
                hand = bool(state.get(key, False))
                values[start + int(hand)*half:start + (int(hand)+1)*half, b] = self.cfg.sensory_gain
        self.current.fill_(self.cfg.tonic_current)
        self.current[self.sensory] = torch.as_tensor(values, device=self.device)
        return self.current

    @torch.no_grad()
    def features(self, states, duration_ms=None):
        self.lif.advance(self.encode(states), self.cfg.step_ms if duration_ms is None else duration_ms)
        return self.lif.readout(self.readout)

    def trace(self, time_ms, index=0):
        groups = {}
        for name in dict.fromkeys(n.cell_type for n in self.conn.neurons):
            ids = [n.idx for n in self.conn.neurons if n.cell_type == name]
            voltage = self.lif.voltage[ids, index]
            groups[name] = {'rate_hz':float(self.lif.rates[ids, index].mean().item() * 1000 / self.cfg.lif_dt_ms),
                            'voltage_mv':float(voltage.mean()),
                            'spikes':int(self.lif.spikes[ids, index].sum())}
        # Sample every modeled population, rather than drawing only the first
        # readout neurons (which would all be visual projection).
        sample=[]
        for name in groups:
            ids=[n.idx for n in self.conn.neurons if n.cell_type==name]
            sample.extend(ids[int(i)] for i in np.linspace(0,len(ids)-1,min(16,len(ids))))
        chosen=torch.as_tensor(sample,device=self.device)
        return {'time_ms':time_ms, 'groups':groups, 'neuron_ids':chosen.cpu().tolist(),
                'rates_hz':(self.lif.rates[chosen,index]*1000/self.cfg.lif_dt_ms).cpu().tolist(),
                'voltage_mv':self.lif.voltage[chosen,index].cpu().tolist()}
