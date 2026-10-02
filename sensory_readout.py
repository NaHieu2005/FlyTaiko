"""Causal measured-neuron sensory/motor readout for a temporal pilot.

Optic-lobe cells are selected by measured incoming photoreceptor contacts.
No chart state, labels, timestamps or rendered RGB pixels enter the decoder.
"""
import numpy as np
from scipy import sparse
import torch

from motor_policy import output_features
from prepare_malecns import ROOT


class SensoryReadout:
    def __init__(self, metadata, arrays, optic_count=1024, root=ROOT):
        photo = np.asarray(arrays['photo'], dtype=np.int64)
        candidates = np.array([i for i, item in enumerate(metadata)
                               if item['group'] == 'ol_intrinsic'], dtype=np.int64)
        contacts = sparse.load_npz(root / 'contacts_pre_post.npz')
        drive = np.asarray(contacts[photo].sum(axis=0)).ravel()
        connected = candidates[drive[candidates] > 0]
        if len(connected) < optic_count:
            raise ValueError(f'Only {len(connected)} optic cells directly connected to photoreceptors')
        rank = np.lexsort((arrays['ids'][connected], -drive[connected]))
        self.optic = connected[rank[:optic_count]]
        self.photo = photo
        self.width = len(photo) + len(self.optic) + 4 * len(arrays['output'])

    def features(self, brain, motor_current, motor_previous):
        device = motor_current.device
        photo = brain.rates[torch.as_tensor(self.photo, device=device)] .T / 100.
        optic = brain.rates[torch.as_tensor(self.optic, device=device)] .T / 100.
        change = (motor_current - motor_previous).clamp(-2, 2)
        return torch.cat((photo, optic, motor_current, change), dim=1).float()
