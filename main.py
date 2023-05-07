from dataclasses import dataclass
from typing import Optional
import warnings
import argparse
import sqlparse
import torch
from torchtext.data.utils import get_tokenizer
import mlflow
from transformer import Transformer
from pipeline import WikiSQL, train


TRAIN_DATA_PATH = 'data/train.csv'
VAL_DATA_PATH = 'data/validation.csv'
TEST_DATA_PATH = 'data/test.csv'
SAVE_PATH = 'saved_models/eng2sql.pt'
MODEL_PATH = 'saved_models/eng2sql.pt'

SPECIAL_TOKENS = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_path', type=str, default=TRAIN_DATA_PATH)
    parser.add_argument('--val_data_path', type=str, default=VAL_DATA_PATH)
    parser.add_argument('--test_data_path', type=str, default=TEST_DATA_PATH)
    parser.add_argument('--save_path', type=str, default=SAVE_PATH)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dmodel', type=int, default=128)
    parser.add_argument('--nstack', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dffn_hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--src_max_len', type=int, default=100)
    parser.add_argument('--tgt_max_len', type=int, default=100)

    return parser.parse_args()


@dataclass
class TransformerConfig:
    d_model: int
    n_stack: int
    n_head: int
    d_ffn_hidden: int
    dropout: float
    src_max_len: int
    tgt_max_len: int
    src_vocab_size: Optional[int] = None
    tgt_vocab_size: Optional[int] = None
    pad_idx: Optional[int] = None

    @classmethod
    def build_from_namespace(cls, args: argparse.Namespace):
        return cls(
            d_model=args.dmodel,
            n_stack=args.nstack,
            n_head=args.nhead,
            d_ffn_hidden=args.dffn_hidden,
            dropout=args.dropout,
            src_max_len=args.src_max_len,
            tgt_max_len=args.tgt_max_len
        )
    
    def to_dict(self):
        return vars(self)
        
    
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    tgt_tokenizer = lambda x: [str(token) for token in sqlparse.parse(x)[0].tokens]
    train_data = WikiSQL(args.train_data_path, src_tokenizer, tgt_tokenizer, SPECIAL_TOKENS)
    val_data = WikiSQL(args.val_data_path, src_tokenizer, tgt_tokenizer, SPECIAL_TOKENS)
    
    config = TransformerConfig.build_from_namespace(args)
    config.src_vocab_size = len(train_data.src_token2idx)
    config.tgt_vocab_size = len(train_data.tgt_token2idx)
    config.pad_idx = SPECIAL_TOKENS['<pad>']
    model = Transformer(config)
    
    mlflow.set_experiment('eng2sql')
    with mlflow.start_run():
        train(
            model,
            train_data,
            val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            path=args.save_path,
            device=args.device
        )

        mlflow.log_params(config.to_dict())
        mlflow.pytorch.log_model(model, 'models')
        
        #mlflow.log_metric('val_loss', ...)
        #mlflow.log_metric('val_acc', ...)
        #mlflow.log_metric('test_loss', ...)
        #mlflow.log_metric('test_acc', ...)
    

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
    
    



