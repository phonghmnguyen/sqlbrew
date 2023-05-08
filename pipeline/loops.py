import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import Adam
from pipeline.lr_scheduler import TransformerScheduledOPT


def train(model, train_data, val_data, epochs=10, batch_size=32, lr=1e-3, weight_decay=1e-4, label_smoothing=0.1, path='', device='cpu'):
    for p in model.parameters():
        if p.dim() > 1:
            init.xavier_uniform_(p)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = TransformerScheduledOPT(optimizer, lr, model.config.d_model, 4000)
    criterion = nn.CrossEntropyLoss(ignore_index=model.config.pad_idx, label_smoothing=label_smoothing)
    model.to(device)
    model.train()
    train_loader = train_data.get_dataloader(batch_size=batch_size, shuffle=True)
    val_loader = val_data.get_dataloader(batch_size=batch_size, shuffle=False)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        for i, batch in enumerate(train_loader):
            src, tgt, tgt_y = batch.src, batch.tgt, batch.tgt_y
            src_mask, tgt_mask = batch.src_mask, batch.tgt_mask

            src, tgt, tgt_y = torch.tensor(src).to(device), torch.tensor(tgt).to(device), torch.tensor(tgt_y).to(device)
            src_mask, tgt_mask = torch.tensor(src_mask).to(device), torch.tensor(tgt_mask).to(device)
        
            output = model(src, tgt, src_mask, tgt_mask)
            flatten_output = output.contiguous().view(-1, output.size(-1))
            flatten_tgt_y = tgt_y.contiguous().view(-1)
            loss = criterion(flatten_output, flatten_tgt_y)
            train_loss += loss.item()
            acc = (flatten_output.argmax(dim=-1) == flatten_tgt_y).sum() / flatten_tgt_y.size(0)
            train_acc += acc.item()
            
            scheduler.zero_grad()
            #optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            #optimizer.step()
           
            
            print(f'Epoch [{epoch + 1}/{epochs}] | Step [{i + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}')
        
        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_acc = 0
            for batch in val_loader:
                src, tgt, tgt_y = batch.src, batch.tgt, batch.tgt_y
                src_mask, tgt_mask = batch.src_mask, batch.tgt_mask
                
                src = torch.tensor(src).to(device)
                tgt = torch.tensor(tgt).to(device)

                output = model(src, tgt, src_mask, tgt_mask)
                flatten_output = output.contiguous().view(-1, output.size(-1))
                flatten_tgt_y = tgt_y.contiguous().view(-1)
                loss = criterion(flatten_output, flatten_tgt_y)
                val_loss += loss.item()
                acc = (flatten_output.argmax(dim=-1) == flatten_tgt_y).sum() / flatten_tgt_y.size(0)
                val_acc += acc.item()
                
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path)

        print(f'Epoch [{epoch + 1}/{epochs}] | Loss: {train_loss / len(train_loader):.4f} | Acc: {train_acc / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_acc / len(val_loader):.4f}')







