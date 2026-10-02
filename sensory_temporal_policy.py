"""Causal 7-mode decoder with memory over measured sensory/motor neurons."""
import torch
from torch import nn


class SensoryTemporalPolicy(nn.Module):
    def __init__(self, width, hidden=128, dropout=.1):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('scale', torch.ones(width))
        self.projection = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.memory = nn.GRU(hidden, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, 7)
        self.dropout = dropout
        self.state = None

    @torch.no_grad()
    def calibrate(self, examples):
        values = torch.as_tensor(examples, dtype=torch.float32, device=self.mean.device)
        self.mean.copy_(values.mean(0))
        self.scale.copy_(values.std(0).clamp_min(.001))

    def reset_state(self):
        self.state = None

    def sequence(self, features, state=None):
        if features.ndim != 3 or features.shape[-1] != self.mean.numel():
            raise ValueError('Expected BxTxF measured-neuron sequence')
        values = ((features - self.mean) / self.scale).clamp(-8, 8)
        values = nn.functional.dropout(values, p=self.dropout, training=self.training)
        output, state = self.memory(self.projection(values), state)
        return self.readout(output), state

    def forward(self, features):
        logits, self.state = self.sequence(features[:, None, :], self.state)
        return logits[:, 0, :]
