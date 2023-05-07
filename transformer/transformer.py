import torch
import torch.nn as nn
from .embedding import Embedding
from .position import PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder
from .generator import Generator


class Transformer(nn.Module):
    """
    The Transformer model composed of an encoder and decoder. The model learns to
    generate a sequence of tokens in the target language from a sequence of tokens
    in the source language.

    Args:
        config: An instance of `TransformerConfig` class which contains the
            configuration settings for the Transformer model.

    Attributes:
        enc_block: A module dictionary that contains the encoder sub-blocks:
            p_emb: A sequential block that contains the embeddings layer and the
                positional encoding layer for the source sequence.
            encoder: An instance of the Encoder class that processes the source sequence.

        dec_block: A module dictionary that contains the decoder sub-blocks:
            p_emb: A sequential block that contains the embeddings layer and the
                positional encoding layer for the target sequence.
            decoder: An instance of the Decoder class that processes the target sequence.
            gen: An instance of the Generator class that maps the output of the decoder
                to a probability distribution over the vocabulary.
    """
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.enc_block = nn.ModuleDict(dict(
            p_emb = nn.Sequential(
                Embedding(config.src_vocab_size, config.d_model, config.pad_idx),
                PositionalEncoding(config.d_model, config.dropout, config.src_max_len)
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
                PositionalEncoding(config.d_model, config.dropout, config.tgt_max_len)
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

    def _encode(self, src, mask):
        """
        Process the source sequence through the encoder and returns the
        output of the encoder.

        Args:
            src: A tensor of shape `(batch_size, src_len)` containing the
                token ids of the source sequences.

        Returns:
            A tensor of shape `(batch_size, src_len, d_model)` representing the
            output of the encoder.
    
        """
        we = self.enc_block.p_emb(src)
        return self.enc_block.encoder(we, mask=mask)
    
    def _decode(self, memory, memory_mask, tgt, mask):
        """
        Process the target sequence through the decoder given the output of the
        encoder, and returns the output of the decoder.

        Args:
            tgt: A tensor of shape `(batch_size, tgt_len)` containing the
                token ids of the target sequences.
            memory: A tensor of shape `(batch_size, src_len, d_model)` representing
                the output of the encoder for the source sequence.
            mask: The look-ahead mask to apply on the target sequence of shape `(batch_size, tgt_len, tgt_len)`.

        Returns:
            A tensor of shape `(batch_size, tgt_len, d_model)` representing the
            output of the decoder.

        """
        we = self.dec_block.p_emb(tgt)
        return self.dec_block.decoder(we, memory, mask=mask, memory_mask=memory_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Process the source and target sequences through the encoder and decoder
        and returns the output of the decoder.

        Args:
            src: The input source sequence of shape `(batch_size, src_len)`.
            tgt: The input target sequence of shape `(batch_size, tgt_len)`.
            mask: The look-ahead mask to apply on the target sequence of shape `(batch_size, tgt_len, tgt_len)`.

        Returns:
            The predicted target sequence of shape `(batch_size, tgt_len, tgt_vocab_size)`.
        """
        enc_out = self._encode(src, src_mask) 
        dec_out = self._decode(enc_out, src_mask, tgt, tgt_mask)
        return self.dec_block.gen(dec_out)

    @classmethod
    def from_pretrained(self, path):
        pass
       
    # inference time
    @torch.no_grad()
    def generate(self, src):
        pass
