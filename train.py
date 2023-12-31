import warnings
import argparse
import torch
from torchtext.data.utils import get_tokenizer
from tokenizers import Tokenizer
import mlflow
from transformer import Transformer, TransformerConfig
from pipeline import WikiSQL, train


TRAIN_DATA_PATH = 'data/train.csv'
VAL_DATA_PATH = 'data/validation.csv'
TEST_DATA_PATH = 'data/test.csv'
SAVE_PATH = 'model/sqlify.pt'
MODEL_PATH = 'model/sqlify.pt'

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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--dmodel', type=int, default=768)
    parser.add_argument('--nstack', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=12)
    parser.add_argument('--dffn_hidden', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--src_max_len', type=int, default=100)
    parser.add_argument('--tgt_max_len', type=int, default=100)

    return parser.parse_args()

        
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    bpe_tokenizer = Tokenizer.from_file('tokenizer/bpetokenizer.json')
    tgt_tokenizer = lambda x: bpe_tokenizer.encode(x).tokens

    train_data = WikiSQL(args.train_data_path, src_tokenizer, tgt_tokenizer, SPECIAL_TOKENS)
    val_data = WikiSQL(args.val_data_path, src_tokenizer, tgt_tokenizer, SPECIAL_TOKENS, False, train_data.src_token2idx, train_data.tgt_token2idx)
    
    config = TransformerConfig(
        d_model=args.dmodel,
        n_stack=args.nstack,
        n_head=args.nhead,
        d_ffn_hidden=args.dffn_hidden,
        dropout=args.dropout,
        src_max_len=args.src_max_len,
        tgt_max_len=args.tgt_max_len,
        src_vocab_size=len(train_data.src_token2idx),
        tgt_vocab_size=len(train_data.tgt_token2idx),
        pad_idx=SPECIAL_TOKENS['<pad>']
    )
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
            label_smoothing=args.label_smoothing,
            path=args.save_path,
            device=args.device
        )

        mlflow.log_params(vars(args))




if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
    
    



