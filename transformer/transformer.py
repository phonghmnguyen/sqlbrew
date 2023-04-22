import torch
import torch.nn as nn
from .embedding import Embedding
from .position import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder
from .generator import Generator


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.enc_block = nn.ModuleDict(dict(
            p_emb = nn.Sequential(
                Embedding(config.src_vocab_size, config.d_model, config.pad_idx),
                PositionalEncoding(config.d_model, config.dropout, config.max_len)
            ),
            encoder = Encoder(
                config.d_model,
                config.n_stack,
                config.n_head,
                config.d_ffn_hidden,
                config.dropout
            )
        ))
        self.dec_block = nn.ModuleDict(dict(
            p_emb = nn.Sequential(
                Embedding(config.tgt_vocab_size, config.d_model, config.pad_idx),
                PositionalEncoding(config.d_model, config.dropout, config.max_len)
            ),
            decoder = Decoder(
                config.d_model,
                config.n_stack,
                config.n_head,
                config.d_ffn_hidden,
                config.dropout
            ),
            gen = Generator(config.d_model, config.tgt_vocab_size),
        ))

    def _encode(self, src):
        we = self.enc_block.p_emb(src)
        return self.enc_block.encoder(we)
    
    def _decoder(self, tgt, memory, mask):
        we = self.dec_block.p_emb(tgt)
        return self.dec_block.decoder(we, memory, mask)
    
    def forward(self, src, tgt, mask):
        enc_out = self._encode(src) 
        dec_out = self._decoder(tgt, enc_out, mask)
        return self.dec_block.gen(dec_out)

    @classmethod
    def load_from_checkpoint(cls, path):
        pass

    @torch.no_grad()
    def generate(self):
        pass
       