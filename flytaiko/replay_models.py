"""Load the published v25 policy independently of training entry points."""
import json
import os
from pathlib import Path

import torch

from flytaiko.sensory_temporal_policy import SensoryTemporalPolicy


def load_v25():
    root = Path(os.environ.get('FLYTAIKO_MODEL_DIR', 'models/v25'))
    config = json.loads((root / 'config.json').read_text())
    checkpoint = root / 'policy.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError('Place the trusted v25 selected checkpoint at ' + str(checkpoint))
    payload = torch.load(checkpoint, map_location='cuda', weights_only=False)
    if payload['config'] != config:
        raise ValueError('v25 checkpoint/config mismatch')
    policy = SensoryTemporalPolicy(payload['feature_width']).cuda()
    policy.load_state_dict(payload['policy'])
    policy.eval()
    return policy, payload, checkpoint, config
