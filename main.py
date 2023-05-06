from dataclasses import dataclass
from torchtext.data.utils import get_tokenizer
from transformer import Transformer
from pipeline import WikiSQL, train


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    pad_idx: int
    d_model: int
    n_stack: int
    n_head: int
    d_ffn_hidden: int
    dropout: float
    src_max_len: int
    tgt_max_len: int


SPECIAL_TOKENS = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}


def main():
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    train_data = WikiSQL('data/train.csv', tokenizer, SPECIAL_TOKENS)
    val_data = WikiSQL('data/validation.csv', tokenizer, SPECIAL_TOKENS)
    # baseline hyperparameters
    config = TransformerConfig(
        src_vocab_size=len(train_data.src_token2idx),
        tgt_vocab_size=len(train_data.tgt_token2idx),
        pad_idx=0,
        d_model=128,
        n_stack=2,
        n_head=4,
        d_ffn_hidden=512,
        dropout=0.1,
        src_max_len=150,
        tgt_max_len=150,
    )
    model = Transformer(config)
    # GO BRUHHHHHHH for a while until running out of memory lol
    #train(model, train_data, val_data, epochs=100, batch_size=32, lr=1e-3, weight_decay=1e-4, device='mps')
    train(model, train_data, val_data, epochs=100, batch_size=32, lr=1e-3, weight_decay=1e-4, device='cpu')



if __name__ == '__main__':
    
    main()

