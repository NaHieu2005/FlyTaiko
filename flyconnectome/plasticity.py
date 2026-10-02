"""
Dopamine-Gated Plasticity for Mushroom Body

Implements the learning rule used in DOOMFLY:
- KC → MBON connections are plastic (can be modified)
- Dopamine neurons (DANs) modulate the plasticity
- When KC is active AND dopamine signal is present → weight changes

This is based on the biological mechanism in Drosophila:
- PAM dopamine neurons: reward signal (appetitive)
- PPL1 dopamine neurons: punishment signal (aversive)
- KC activity + DAN activity → MBON synapse depression/potentiation

For Taiko:
- Great/Good hit → reward dopamine → strengthen correct associations
- Miss/Wrong → punishment dopamine → weaken incorrect associations

Reference: Aso & Rubin (2016), Mushroom body output neurons encode
valence and guide memory-based action selection in Drosophila
"""

import torch
import numpy as np
from scipy import sparse
from typing import Optional


class DopaminePlasticity:
    """Dopamine-gated plasticity rule for KC→MBON connections.

    The rule follows DOOMFLY's approach:
    ΔW_ij = η * DA * KC_i * MBON_j_eligibility

    Where:
    - η: learning rate
    - DA: dopamine signal (positive for reward, negative for punishment)
    - KC_i: Kenyon Cell activity (pre-synaptic)
    - MBON_j_eligibility: eligibility trace at MBON (post-synaptic)

    Eligibility traces allow temporal credit assignment:
    the brain remembers which connections were recently active
    and modifies them when dopamine arrives.
    """

    def __init__(
        self,
        num_neurons: int,
        kc_ids: np.ndarray,
        mbon_ids: np.ndarray,
        dan_ids: np.ndarray,
        kc_mbon_mask: sparse.csr_matrix,
        learning_rate: float = 0.001,
        eligibility_decay: float = 0.95,   # Per-step decay of eligibility trace
        weight_min: float = -2.0,          # Minimum weight
        weight_max: float = 2.0,           # Maximum weight
        dopamine_decay: float = 0.9,       # Decay of dopamine signal
        device: str = 'auto',
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.N = num_neurons
        self.kc_ids = torch.tensor(kc_ids, dtype=torch.long, device=self.device)
        self.mbon_ids = torch.tensor(mbon_ids, dtype=torch.long, device=self.device)
        self.dan_ids = torch.tensor(dan_ids, dtype=torch.long, device=self.device)

        self.lr = learning_rate
        self.elig_decay = eligibility_decay
        self.w_min = weight_min
        self.w_max = weight_max
        self.da_decay = dopamine_decay

        # Build KC→MBON connection indices from mask
        coo = kc_mbon_mask.tocoo()
        self.plastic_rows = torch.tensor(coo.row, dtype=torch.long,
                                         device=self.device)
        self.plastic_cols = torch.tensor(coo.col, dtype=torch.long,
                                         device=self.device)
        self.num_plastic = len(self.plastic_rows)

        # Current plastic weights (stored separately for efficient updates)
        self.plastic_weights = torch.tensor(
            coo.data, dtype=torch.float32, device=self.device
        )
        # Initial weights (for reference)
        self.initial_weights = self.plastic_weights.clone()

        # Eligibility traces for each plastic connection
        self.eligibility = torch.zeros(self.num_plastic, device=self.device)

        # Dopamine level (decays over time)
        self.dopamine_level = torch.zeros(1, device=self.device)

        # Stats
        self.total_updates = 0
        self.total_reward = 0.0
        self.weight_change_history = []

    def reset(self, reset_weights: bool = False):
        """Reset plasticity state (but optionally keep learned weights)."""
        self.eligibility.zero_()
        self.dopamine_level.zero_()
        if reset_weights:
            self.plastic_weights = self.initial_weights.clone()
        self.total_updates = 0
        self.total_reward = 0.0
        self.weight_change_history = []

    @torch.no_grad()
    def update_eligibility(self, spikes: torch.Tensor):
        """Update eligibility traces based on current neural activity.

        Called every LIF simulation step. Marks connections where
        the pre-synaptic KC and post-synaptic MBON were co-active.

        Args:
            spikes: (N,) binary spike vector from LIF network
        """
        # Decay existing eligibility
        self.eligibility *= self.elig_decay

        # KC activity at pre-synaptic side
        kc_spikes = spikes[self.plastic_rows]
        # MBON activity at post-synaptic side (use voltage/rate as proxy)
        mbon_spikes = spikes[self.plastic_cols]

        # Hebbian-like: increase eligibility where pre AND post are active
        # Also includes pre-only trace (for anti-Hebbian learning)
        co_active = kc_spikes * mbon_spikes
        pre_only = kc_spikes * (1.0 - mbon_spikes) * 0.3  # Weaker pre-only trace

        self.eligibility += co_active + pre_only

    @torch.no_grad()
    def deliver_dopamine(self, reward: float):
        """Deliver a dopamine reward/punishment signal.

        Called when the game gives feedback (hit/miss).

        Args:
            reward: Positive for reward (good hit), negative for punishment (miss)
                    Typical range: -1.0 to +1.0
        """
        self.dopamine_level += reward
        self.total_reward += reward

    @torch.no_grad()
    def apply_plasticity(self) -> float:
        """Apply the dopamine-gated plasticity rule.

        Should be called periodically (e.g., every game step).

        Returns:
            Total absolute weight change applied
        """
        # Decay dopamine
        da = self.dopamine_level.item()
        self.dopamine_level *= self.da_decay

        if abs(da) < 1e-6:
            return 0.0

        # Weight update: ΔW = η * DA * eligibility
        # Reward (DA > 0) → strengthen eligible connections
        # Punishment (DA < 0) → weaken eligible connections
        delta_w = self.lr * da * self.eligibility

        # Apply update
        self.plastic_weights += delta_w

        # Clamp weights
        self.plastic_weights.clamp_(self.w_min, self.w_max)

        # Stats
        total_change = delta_w.abs().sum().item()
        self.total_updates += 1
        self.weight_change_history.append(total_change)

        return total_change

    def get_weight_update_sparse(self) -> torch.Tensor:
        """Get the current plastic weights as a sparse delta tensor.

        Used to update the LIF network's weight matrix.

        Returns:
            Sparse tensor representing the current KC→MBON weights
        """
        indices = torch.stack([self.plastic_rows, self.plastic_cols])
        return torch.sparse_coo_tensor(
            indices, self.plastic_weights,
            size=(self.N, self.N),
            device=self.device
        ).coalesce()

    def get_stats(self) -> dict:
        """Get plasticity statistics."""
        w = self.plastic_weights
        w_init = self.initial_weights
        diff = (w - w_init)

        return {
            'num_plastic_connections': self.num_plastic,
            'total_updates': self.total_updates,
            'total_reward': self.total_reward,
            'weight_mean': w.mean().item(),
            'weight_std': w.std().item(),
            'weight_min': w.min().item(),
            'weight_max': w.max().item(),
            'mean_change_from_initial': diff.abs().mean().item(),
            'max_change_from_initial': diff.abs().max().item(),
            'frac_strengthened': (diff > 0.01).float().mean().item(),
            'frac_weakened': (diff < -0.01).float().mean().item(),
        }

    def get_learning_curve(self) -> list:
        """Get the history of weight changes for plotting."""
        return self.weight_change_history


if __name__ == '__main__':
    # Quick test
    from loader import generate_synthetic_connectome

    data = generate_synthetic_connectome(num_neurons=5000)
    plasticity = DopaminePlasticity(
        num_neurons=data.num_neurons,
        kc_ids=data.kc_ids,
        mbon_ids=data.mbon_ids,
        dan_ids=data.dan_ids,
        kc_mbon_mask=data.kc_mbon_mask,
        device='cpu'
    )

    print(f"Plastic connections: {plasticity.num_plastic}")

    # Simulate some activity
    fake_spikes = torch.zeros(data.num_neurons)
    fake_spikes[data.kc_ids[:100]] = 1.0
    fake_spikes[data.mbon_ids[:10]] = 1.0

    # Update eligibility a few times
    for _ in range(10):
        plasticity.update_eligibility(fake_spikes)

    # Deliver reward and apply
    plasticity.deliver_dopamine(1.0)
    change = plasticity.apply_plasticity()
    print(f"Weight change after reward: {change:.6f}")

    # Deliver punishment
    plasticity.deliver_dopamine(-1.0)
    change = plasticity.apply_plasticity()
    print(f"Weight change after punishment: {change:.6f}")

    print(plasticity.get_stats())
