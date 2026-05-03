import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from utils.metrics import evaluate_predictions

class GRU4RecNetwork(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.fc = nn.Linear(hidden_size, num_items)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.gru(emb)
        return self.fc(out[:, -1, :])

class GRU4RecAgent:
    def __init__(self, num_items, max_seq_len, embedding_dim=64, hidden_size=128, dropout=0.2, lr=0.001, device='cpu'):
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.device = device
        self.lr = lr
        
        self.model = GRU4RecNetwork(num_items, embedding_dim, hidden_size, dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, top_k=10, ui_callback=None):
        train_data = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'val_metrics': []}
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
                
            avg_loss = total_loss / len(train_data)
            history['train_loss'].append(avg_loss)
            
            val_metrics = self.evaluate(X_val, y_val, batch_size, top_k)
            history['val_metrics'].append(val_metrics)
            
            if ui_callback:
                ui_callback(epoch, epochs, avg_loss, val_metrics)
                
        return history
        
    def predict(self, seq, top_k=10):
        self.model.eval()
        with torch.no_grad():
            x = torch.LongTensor([seq]).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits[0], dim=-1)
            probs[0] = 0.0 # Ignore padding
            top_probs, top_indices = torch.topk(probs, top_k)
            return top_indices.cpu().numpy(), top_probs.cpu().numpy()
            
    def evaluate(self, X_val, y_val, batch_size=256, top_k=10):
        self.model.eval()
        val_data = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                all_preds.extend(logits.cpu().numpy())
                
        return evaluate_predictions(np.array(all_preds), y_val, k=top_k)
        
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            return True
        return False
