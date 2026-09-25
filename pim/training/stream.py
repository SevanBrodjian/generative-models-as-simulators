"""BlockStream: a shuffled-block reader over a flat memmap, with one-block prefetch.

Every sequence in the corpus comes from its own seed, so a contiguous block is a random sample:
blocks are read in shuffled order, shuffled within, and the next block is prefetched on a thread.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import torch


class BlockStream:
    """Yields (batch, ...) torch tensors from ``obs[lo:hi]`` forever."""

    def __init__(self, obs, lo: int, hi: int, batch: int, block: int, seed: int,
                 shuffle: bool = True):
        self.obs, self.lo, self.hi = obs, lo, hi
        self.batch, self.block, self.shuffle = batch, block, shuffle
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue = queue.Queue(maxsize=2)
        self.starts = np.arange(lo, hi - block + 1, block)
        if len(self.starts) == 0:
            raise ValueError(f"range [{lo:,}, {hi:,}) is shorter than one {block:,}-sequence block")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            order = self.rng.permutation(self.starts) if self.shuffle else self.starts
            for s in order:
                blk = np.array(self.obs[s: s + self.block])
                if self.shuffle:
                    blk = blk[self.rng.permutation(len(blk))]
                self.q.put(blk)

    def batches(self):
        while True:
            blk = self.q.get()
            for i in range(0, len(blk) - self.batch + 1, self.batch):
                yield torch.from_numpy(blk[i: i + self.batch])
