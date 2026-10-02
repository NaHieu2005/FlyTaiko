"""Fixed-tree CSR accumulation without CUDA atomics or invented graph edges.

One program owns each postsynaptic row. Fixed chunks and fixed batch padding
keep reduction order independent of the actual number of simulated games.
Reproducibility is tested on the deployed hardware, not claimed cross-platform.
"""
import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    @triton.jit
    def _csr_rows(CROW, COL, WEIGHT, INPUT, ROWS, OUTPUT,
                  BATCH: tl.constexpr, PAD_BATCH: tl.constexpr,
                  CHUNK: tl.constexpr):
        row = tl.load(ROWS + tl.program_id(0))
        start = tl.load(CROW + row)
        end = tl.load(CROW + row + 1)
        lanes = tl.arange(0, CHUNK)
        columns = tl.arange(0, PAD_BATCH)
        total = tl.full((PAD_BATCH,), 0., tl.float32)
        for offset in range(start, end, CHUNK):
            edge = offset + lanes
            pre = tl.load(COL + edge, edge < end, other=0)
            weight = tl.load(WEIGHT + edge, edge < end, other=0)
            spikes = tl.load(INPUT + pre[:, None] * BATCH + columns[None, :],
                             (edge[:, None] < end) & (columns[None, :] < BATCH), other=0.)
            contribution = weight[:, None] * spikes
            total += tl.sum(contribution, axis=0)
        tl.store(OUTPUT + row * BATCH + columns, total, columns < BATCH)


class FixedCSR:
    def __init__(self, matrix, device, padded_batch=8):
        if triton is None:
            raise RuntimeError('Deterministic GPU synapses require Triton; no nondeterministic fallback')
        self.rows, self.columns = matrix.shape
        self.padded_batch = padded_batch
        self.crow = torch.as_tensor(matrix.indptr.astype(np.int32), device=device)
        self.col = torch.as_tensor(matrix.indices.astype(np.int32), device=device)
        self.weight = torch.as_tensor(matrix.data, dtype=torch.float32, device=device)
        counts = np.diff(matrix.indptr)
        self.groups = []
        lower = -1
        for chunk in (32, 128, 512):
            selected = np.flatnonzero((counts > lower) & (counts <= chunk if chunk < 512 else True))
            if len(selected):
                self.groups.append((torch.as_tensor(selected.astype(np.int32), device=device), chunk))
            lower = chunk

    def __call__(self, spikes):
        if spikes.shape[0] != self.columns or spikes.shape[1] > self.padded_batch:
            raise ValueError('Unexpected CSR input shape')
        spikes = spikes.contiguous()
        output = torch.empty((self.rows,spikes.shape[1]),dtype=spikes.dtype,device=spikes.device)
        for rows, chunk in self.groups:
            _csr_rows[(len(rows),)](self.crow, self.col, self.weight, spikes, rows, output,
                                   spikes.shape[1], self.padded_batch, chunk,
                                   num_warps=4, enable_fp_fusion=False)
        return output
