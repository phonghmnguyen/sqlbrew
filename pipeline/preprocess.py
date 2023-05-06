import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pipeline.batch import Batch

class WikiSQL(Dataset):
    def __init__(self, data_path, tokenizer, special_token_map=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        self.special_token_map = special_token_map
        self.src_token2idx = self.tgt_token2idx = special_token_map # (token, idx)
        self.src_idx2token = self.tgt_idx2token = dict((idx, token) for token, idx in self.special_token_map.items()) # (idx, token)
        self.data = self.load_data()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _build_vocab(self, src, tgt):
        src = ['<sos>'] + src + ['<eos>']
        for token in src:
            if token not in self.src_token2idx:
                self.src_token2idx[token] = len(self.src_token2idx)
                self.src_idx2token[self.src_token2idx[token]] = token
        
        tgt = ['<sos>'] + tgt + ['<eos>']
        for token in tgt:
            if token not in self.tgt_token2idx:
                self.tgt_token2idx[token] = len(self.tgt_token2idx)
                self.tgt_idx2token[self.tgt_token2idx[token]] = token

    
    def load_data(self):
        data = []
        with open(self.data_path, 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                data.append(line)
                self._build_vocab(self.tokenizer(line['question']), self.tokenizer(line['sql']))
        return data
    
    def collate_fn(self, batch):
        src_tokens = [self.tokenizer(item['question']) for item in batch]
        tgt_tokens = [self.tokenizer(item['sql']) for item in batch]
        src_tokens = [[self.src_token2idx[token] for token in tokens] for tokens in src_tokens]
        tgt_tokens = [[self.tgt_token2idx[token] for token in tokens] for tokens in tgt_tokens]
        src_tensor = pad_sequence([torch.tensor(tokens) for tokens in src_tokens], batch_first=True)
        tgt_tensor = pad_sequence([torch.tensor(tokens) for tokens in tgt_tokens], batch_first=True)
        return Batch(src_tensor, tgt_tensor)
    
    def get_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
    
    def get_batch(self, batch_size, shuffle=True):
        dataloader = self.get_dataloader(batch_size, shuffle)
        for batch in dataloader:
            yield batch

