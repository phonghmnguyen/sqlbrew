import torch
import numpy as np

class Batch:
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        # padding mask
        self.src_mask = (src != pad).unsqueeze(-2)
        self._size = src.size(0)

        if tgt is not None:
            # shift target sequence for teacher forcing
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]

            self.tgt_mask = (self.tgt != pad).unsqueeze(-2) \
            & self._subsequent_mask(self.tgt.size(-1)).type(torch.bool)

    def _subsequent_mask(self, size):
        """
            Make a look-ahead mask for subsequent positions.
        """

        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
        

       