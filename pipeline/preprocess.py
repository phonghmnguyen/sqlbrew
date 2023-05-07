import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Batch:
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        # padding mask
        self.src_mask = (src != pad).unsqueeze(-2)
        self.size = src.size(0)

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
    
    def __len__(self):
        return self.size
        

class WikiSQL(Dataset):
    def __init__(self, data_path, src_tokenizer, tgt_tokenizer, special_token_map=None):
        self.data_path = data_path
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
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
                self._build_vocab(self.src_tokenizer(line['question']), self.tgt_tokenizer(line['sql']))
        return data
    
    def collate_fn(self, batch):
        src_tokens = [self.src_tokenizer(item['question']) for item in batch]
        tgt_tokens = [self.tgt_tokenizer(item['sql']) for item in batch]
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

