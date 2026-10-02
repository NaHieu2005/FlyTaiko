"""
LIF Spiking Neural Network on GPU

Implements a Leaky Integrate-and-Fire network using the connectome
adjacency matrix. Runs on GPU via PyTorch sparse tensor operations.

Based on the approach from DOOMFLY:
- τ * dV/dt = -(V - V_rest) + R * I_syn + I_ext
- Spike when V >= V_thresh, then V = V_reset for refractory period
- I_syn = W @ spike_vector (sparse matmul)

Key differences from standard LIF:
- Uses the connectome's signed weights (excitatory/inhibitory)
- Includes refractory period
- Batched computation for GPU efficiency
"""

import torch
import numpy as np
from scipy import sparse
from typing import Optional, Tuple


class LIFNetwork:
    """GPU-accelerated Leaky Integrate-and-Fire spiking neural network.

    Uses the fly connectome adjacency matrix as the connection weights.
    All neuron dynamics are computed on GPU using PyTorch sparse tensors.
    """

    def __init__(
        self,
        adjacency: sparse.csr_matrix,
        dt: float = 0.5,          # Simulation timestep (ms)
        tau: float = 10.0,        # Membrane time constant (ms)
        v_rest: float = -65.0,    # Resting potential (mV)
        v_thresh: float = -50.0,  # Spike threshold (mV)
        v_reset: float = -65.0,   # Reset potential (mV)
        r_membrane: float = 1.0,  # Membrane resistance
        refractory_ms: float = 2.0,  # Refractory period (ms)
        spectral_scale: float = 0.9,  # Scale factor for weight matrix
        device: str = 'auto',
    ):
        """
        Args:
            adjacency: (N, N) sparse adjacency matrix from connectome
            dt: Simulation timestep in ms
            tau: Membrane time constant in ms
            v_rest: Resting membrane potential in mV
            v_thresh: Spike threshold in mV
            v_reset: Reset potential after spike in mV
            r_membrane: Membrane resistance (scaling factor for input)
            refractory_ms: Duration of refractory period in ms
            spectral_scale: Global scale factor to prevent blowup
            device: 'cuda', 'cpu', or 'auto'
        """
        # Device selection
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.N = adjacency.shape[0]
        self.dt = dt
        self.tau = tau
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.r_membrane = r_membrane
        self.refractory_steps = int(refractory_ms / dt)

        # Convert adjacency to PyTorch sparse tensor on GPU
        self.W = self._build_weight_matrix(adjacency, spectral_scale)

        # State variables
        self.voltage = torch.full((self.N,), v_rest, device=self.device)
        self.spikes = torch.zeros(self.N, device=self.device)
        self.refractory_counter = torch.zeros(self.N, dtype=torch.int32,
                                              device=self.device)

        # Spike history for readout (rolling window)
        self._spike_history_len = 30  # ~15ms at dt=0.5ms
        self._spike_history = torch.zeros(
            (self._spike_history_len, self.N), device=self.device
        )
        self._history_idx = 0

        # Tracking
        self.total_steps = 0
        self.total_spikes = 0

    def _build_weight_matrix(self, adj: sparse.csr_matrix,
                             scale: float) -> torch.Tensor:
        """Convert scipy sparse matrix to PyTorch sparse tensor.

        Applies a global scaling factor (like DOOM-x-Fly's "one global
        scale factor so the network doesn't blow up or go silent").
        """
        # Convert to COO format
        coo = adj.tocoo().astype(np.float32)

        # Estimate spectral radius for scaling
        try:
            from scipy.sparse.linalg import eigs
            eigenvalues = eigs(adj.astype(np.float64), k=1,
                               which='LM', return_eigenvectors=False)
            spectral_radius = np.abs(eigenvalues[0])
        except Exception:
            # Fallback: approximate with Frobenius norm
            spectral_radius = sparse.linalg.norm(adj, 'fro') / np.sqrt(self.N)
            spectral_radius = max(spectral_radius, 1e-6)

        # Scale weights
        global_scale = scale / max(spectral_radius, 1e-6)

        indices = torch.tensor(
            np.vstack([coo.row, coo.col]),
            dtype=torch.long,
            device=self.device
        )
        values = torch.tensor(
            coo.data * global_scale,
            dtype=torch.float32,
            device=self.device
        )

        W = torch.sparse_coo_tensor(
            indices, values, size=(self.N, self.N),
            device=self.device
        ).coalesce()

        return W

    def reset(self):
        """Reset all state to initial conditions."""
        self.voltage.fill_(self.v_rest)
        self.spikes.zero_()
        self.refractory_counter.zero_()
        self._spike_history.zero_()
        self._history_idx = 0
        self.total_steps = 0
        self.total_spikes = 0

    @torch.no_grad()
    def step(self, external_current: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run one simulation timestep.

        Args:
            external_current: (N,) tensor of external input currents.
                              Only non-zero for sensory neurons typically.

        Returns:
            spikes: (N,) binary tensor of which neurons fired
        """
        # 1. Compute synaptic current from spikes
        #    I_syn = W @ spikes (sparse matmul)
        i_syn = torch.mv(self.W, self.spikes)

        # 2. Add external current
        if external_current is not None:
            i_total = i_syn + external_current
        else:
            i_total = i_syn

        # 3. LIF voltage update (Euler method)
        #    τ * dV/dt = -(V - V_rest) + R * I
        #    V_new = V + dt/τ * (-(V - V_rest) + R * I)
        dv = (self.dt / self.tau) * (
            -(self.voltage - self.v_rest) + self.r_membrane * i_total
        )
        self.voltage += dv

        # 4. Refractory: neurons in refractory period don't update
        refractory_mask = self.refractory_counter > 0
        self.voltage[refractory_mask] = self.v_reset
        self.refractory_counter[refractory_mask] -= 1

        # 5. Spike detection
        self.spikes = (self.voltage >= self.v_thresh).float()

        # 6. Reset spiking neurons
        spiked = self.spikes.bool()
        self.voltage[spiked] = self.v_reset
        self.refractory_counter[spiked] = self.refractory_steps

        # 7. Update spike history
        self._spike_history[self._history_idx] = self.spikes
        self._history_idx = (self._history_idx + 1) % self._spike_history_len

        # 8. Track stats
        self.total_steps += 1
        self.total_spikes += self.spikes.sum().item()

        return self.spikes

    def run_steps(self, n_steps: int,
                  external_current: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """Run multiple simulation steps.

        Args:
            n_steps: Number of steps to run
            external_current: (N,) constant external current for all steps

        Returns:
            firing_rates: (N,) average firing rate over the steps
        """
        total_spikes = torch.zeros(self.N, device=self.device)
        for _ in range(n_steps):
            spikes = self.step(external_current)
            total_spikes += spikes
        return total_spikes / n_steps

    def get_firing_rates(self, neuron_ids: Optional[np.ndarray] = None
                         ) -> torch.Tensor:
        """Get recent firing rates from spike history.

        Args:
            neuron_ids: Indices of neurons to read. None = all neurons.

        Returns:
            Firing rates (spikes per timestep) for requested neurons
        """
        rates = self._spike_history.mean(dim=0)
        if neuron_ids is not None:
            return rates[neuron_ids]
        return rates

    def get_state_vector(self, neuron_ids: np.ndarray) -> torch.Tensor:
        """Get a combined state vector for readout.

        Combines voltage (normalized) and firing rate for richer signal.

        Args:
            neuron_ids: Indices of readout neurons

        Returns:
            (2 * len(neuron_ids),) tensor: [normalized_voltages, firing_rates]
        """
        # Normalized voltage (0 = rest, 1 = threshold)
        v_range = self.v_thresh - self.v_rest
        v_norm = (self.voltage[neuron_ids] - self.v_rest) / v_range
        v_norm = v_norm.clamp(0.0, 1.5)

        # Firing rates
        rates = self.get_firing_rates(neuron_ids)

        return torch.cat([v_norm, rates])

    def inject_current(self, neuron_ids: np.ndarray,
                       values: torch.Tensor) -> torch.Tensor:
        """Create an external current vector with values at specified neurons.

        Args:
            neuron_ids: Indices of neurons to inject current into
            values: Current values for those neurons

        Returns:
            (N,) external current tensor
        """
        current = torch.zeros(self.N, device=self.device)
        current[neuron_ids] = values.to(self.device)
        return current

    def get_stats(self) -> dict:
        """Get simulation statistics."""
        mean_rate = self.total_spikes / max(self.total_steps * self.N, 1)
        active = (self.get_firing_rates() > 0).sum().item()
        return {
            'total_steps': self.total_steps,
            'total_spikes': int(self.total_spikes),
            'mean_firing_rate': mean_rate,
            'active_neurons': active,
            'active_fraction': active / self.N,
            'voltage_mean': self.voltage.mean().item(),
            'voltage_std': self.voltage.std().item(),
            'has_nan': torch.isnan(self.voltage).any().item(),
        }

    def update_weights(self, delta_W: torch.Tensor,
                       mask: Optional[torch.Tensor] = None):
        """Apply weight changes (for plasticity).

        Args:
            delta_W: Sparse tensor of weight changes
            mask: Optional mask to restrict which connections change
        """
        if mask is not None:
            delta_W = delta_W * mask
        # Note: adding sparse tensors
        self.W = (self.W + delta_W).coalesce()


def create_lif_network(adjacency: sparse.csr_matrix,
                       device: str = 'auto',
                       **kwargs) -> LIFNetwork:
    """Convenience function to create a LIF network from connectome data.

    Args:
        adjacency: Connectome adjacency matrix
        device: 'cuda', 'cpu', or 'auto'
        **kwargs: Additional LIF parameters

    Returns:
        Configured LIFNetwork
    """
    return LIFNetwork(adjacency, device=device, **kwargs)


if __name__ == '__main__':
    from loader import generate_synthetic_connectome

    print("Generating connectome...")
    data = generate_synthetic_connectome(num_neurons=5000)
    print(data.summary())

    print("\nCreating LIF network...")
    net = create_lif_network(data.adjacency, device='cpu')
    print(f"Device: {net.device}")
    print(f"Neurons: {net.N:,}")

    print("\nRunning 100 steps...")
    current = net.inject_current(
        data.sensory_ids[:100],
        torch.ones(100) * 5.0
    )
    for i in range(100):
        net.step(current)

    stats = net.get_stats()
    print(f"Stats: {stats}")
