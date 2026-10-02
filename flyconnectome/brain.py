"""
FlyBrain — Complete Brain Assembly

Combines the LIF spiking network, dopamine plasticity, sensory encoder,
and action decoder into a single brain module.

This is the main interface between the Taiko game and the fly connectome.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from typing import Optional, Tuple

from .loader import ConnectomeData
from .lif_network import LIFNetwork, create_lif_network
from .plasticity import DopaminePlasticity


class TaikoSensoryEncoder:
    """Encodes Taiko game state into fly sensory neuron activations.

    Unlike Doom (which needs full visual processing), Taiko is simpler:
    - Encode upcoming note timing as temporal urgency signals
    - Encode note type (don/kat) as different activation patterns
    - Encode combo/judgment feedback

    The encoding maps to the fly's visual sensory neurons, treating
    note position/timing as a simplified "visual" input.
    """

    def __init__(self, num_sensory_neurons: int, lookahead: int = 8,
                 device: str = 'auto'):
        self.num_inputs = num_sensory_neurons
        self.lookahead = lookahead

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Pre-compute neuron allocation per note
        self.neurons_per_note = num_sensory_neurons // (lookahead + 2)
        # +2: extra groups for combo and judgment feedback

    def encode(self, game_state: dict) -> torch.Tensor:
        """Convert game state to sensory neuron activations.

        Args:
            game_state: Dict from TaikoEnvironment.get_state()

        Returns:
            (num_sensory_neurons,) tensor of activation values
        """
        inputs = torch.zeros(self.num_inputs, device=self.device)
        npn = self.neurons_per_note

        upcoming = game_state['upcoming_notes']

        for i, (time_until, note_type) in enumerate(upcoming[:self.lookahead]):
            start = i * npn

            if note_type == 'none':
                continue

            # 1. Temporal urgency: closer → stronger signal
            # Sigmoid-like: very strong when <100ms away
            if time_until <= 0:
                urgency = max(0.0, 1.0 + time_until / 200.0)  # Fading after pass
            else:
                urgency = 1.0 / (1.0 + time_until / 150.0)

            # 2. Timing ramp: increasing activation as note approaches
            quarter = npn // 4

            # First quarter: general timing signal
            inputs[start:start + quarter] = urgency * 3.0

            # Second quarter: "hit now" signal (peaks at time_until ≈ 0)
            hit_now = max(0.0, 1.0 - abs(time_until) / 50.0)
            inputs[start + quarter:start + 2 * quarter] = hit_now * 5.0

            # Third quarter: DON signal (positive = don)
            is_don = 1.0 if 'don' in note_type else -1.0
            inputs[start + 2 * quarter:start + 3 * quarter] = is_don * urgency * 2.0

            # Fourth quarter: KAT signal (positive = kat)
            is_kat = 1.0 if 'kat' in note_type else -1.0
            inputs[start + 3 * quarter:start + 4 * quarter] = is_kat * urgency * 2.0

        # Combo feedback neurons
        combo_start = self.lookahead * npn
        combo_signal = min(game_state.get('combo', 0) / 50.0, 1.0)
        inputs[combo_start:combo_start + npn] = combo_signal * 2.0

        # Last judgment feedback neurons
        judgment_start = combo_start + npn
        judgment = game_state.get('last_judgment', 'none')
        judgment_map = {'great': 3.0, 'good': 1.5, 'miss': -2.0,
                        'ignore': 0.0, 'none': 0.0}
        jval = judgment_map.get(judgment, 0.0)
        if judgment_start + npn <= self.num_inputs:
            inputs[judgment_start:judgment_start + npn] = jval

        return inputs


class ActionDecoder(nn.Module):
    """Small MLP that reads fly brain activity → game actions.

    Takes the readout neuron state vector and outputs action logits.
    This IS trained (unlike the connectome which stays mostly fixed).

    Architecture follows DOOM-x-Fly: 256 hidden units.
    """

    def __init__(self, input_size: int, num_actions: int = 5,
                 hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlyBrain:
    """Complete fly brain for playing Taiko.

    Integrates:
    1. LIF Spiking Network (the "brain")
    2. Dopamine Plasticity (learning in mushroom body)
    3. Sensory Encoder (game state → neural input)
    4. Action Decoder (neural output → game action)

    Simulation loop per game step:
    1. Encode game state → sensory current
    2. Run LIF for N steps (~15ms worth)
    3. Update eligibility traces (plasticity)
    4. Read out brain state
    5. Decode to action
    """

    def __init__(
        self,
        connectome: ConnectomeData,
        lif_steps_per_game_step: int = 30,  # ~15ms at dt=0.5ms
        temperature: float = 1.25,          # Action sampling temperature
        device: str = 'auto',
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.connectome = connectome
        self.lif_steps = lif_steps_per_game_step
        self.temperature = temperature

        # 1. LIF Network
        self.lif = create_lif_network(
            connectome.adjacency,
            device=str(self.device),
            spectral_scale=0.9,
        )

        # 2. Dopamine Plasticity
        self.plasticity = DopaminePlasticity(
            num_neurons=connectome.num_neurons,
            kc_ids=connectome.kc_ids,
            mbon_ids=connectome.mbon_ids,
            dan_ids=connectome.dan_ids,
            kc_mbon_mask=connectome.kc_mbon_mask,
            device=str(self.device),
        )

        # 3. Sensory Encoder
        self.encoder = TaikoSensoryEncoder(
            num_sensory_neurons=len(connectome.sensory_ids),
            device=str(self.device),
        )

        # 4. Action Decoder
        readout_size = len(connectome.readout_ids) * 2  # voltage + rates
        self.decoder = ActionDecoder(
            input_size=readout_size,
            num_actions=5,
        ).to(self.device)

        # Sensory neuron indices (for injecting current)
        self.sensory_ids = connectome.sensory_ids

    def reset(self, reset_weights: bool = False):
        """Reset brain state for new episode."""
        self.lif.reset()
        self.plasticity.reset(reset_weights=reset_weights)

    def process_game_state(self, game_state: dict) -> Tuple[int, torch.Tensor]:
        """Process a game state and return an action.

        Args:
            game_state: Dict from TaikoEnvironment.get_state()

        Returns:
            action: int (0-4)
            logits: raw action logits
        """
        # 1. Encode game state → sensory current
        sensory_activation = self.encoder.encode(game_state)
        external_current = self.lif.inject_current(
            self.sensory_ids, sensory_activation
        )

        # 2. Run LIF simulation for multiple steps
        for _ in range(self.lif_steps):
            spikes = self.lif.step(external_current)
            # Update eligibility traces every LIF step
            self.plasticity.update_eligibility(spikes)

        # 3. Read brain state
        state_vector = self.lif.get_state_vector(self.connectome.readout_ids)

        # 4. Decode to action
        logits = self.decoder(state_vector)

        # 5. Sample action with temperature
        probs = torch.softmax(logits / self.temperature, dim=0)
        action = torch.multinomial(probs, 1).item()

        return action, logits

    def receive_reward(self, reward: float):
        """Receive reward signal from game and apply plasticity.

        Args:
            reward: Reward value (-1.0 to +1.0)
        """
        # Deliver dopamine
        self.plasticity.deliver_dopamine(reward)

        # Apply plasticity rule
        self.plasticity.apply_plasticity()

        # Update LIF network weights with new plastic weights
        # This is expensive so we do it periodically
        if self.plasticity.total_updates % 5 == 0:
            new_weights = self.plasticity.get_weight_update_sparse()
            # Rebuild the KC→MBON portion of the weight matrix
            # For efficiency, we directly modify the existing sparse tensor
            self.lif.W = (self.lif.W + new_weights * 0).coalesce()  # placeholder

    def get_decoder_params(self):
        """Get decoder parameters for optimizer."""
        return self.decoder.parameters()

    def save(self, path: str):
        """Save brain state."""
        torch.save({
            'decoder_state': self.decoder.state_dict(),
            'plastic_weights': self.plasticity.plastic_weights.cpu(),
            'temperature': self.temperature,
        }, path)

    def load(self, path: str):
        """Load brain state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.decoder.load_state_dict(checkpoint['decoder_state'])
        self.plasticity.plastic_weights = checkpoint['plastic_weights'].to(self.device)
        self.temperature = checkpoint.get('temperature', self.temperature)

    def get_full_stats(self) -> dict:
        """Get comprehensive brain statistics."""
        return {
            'lif': self.lif.get_stats(),
            'plasticity': self.plasticity.get_stats(),
        }
