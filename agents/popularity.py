import numpy as np
import pickle
import os
from utils.metrics import evaluate_predictions

class PopularityAgent:
    def __init__(self, num_items, max_seq_len=None):
        self.num_items = num_items
        self.item_counts = np.zeros(num_items)
        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=1, batch_size=None, top_k=10, ui_callback=None):
        # Simply count frequencies in training data
        for y in y_train:
            self.item_counts[y] += 1
            
        history = {'train_loss': [0.0], 'val_metrics': []}
        
        # Evaluate once since it's not iterative
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val, top_k=top_k)
            history['val_metrics'].append(val_metrics)
            
            if ui_callback:
                ui_callback(1, 1, 0.0, val_metrics)
                
        return history
        
    def predict(self, seq=None, top_k=10):
        # Always recommend the most popular items globally
        probs = self.item_counts.copy()
        probs[0] = 0.0 # ignore padding
        
        top_indices = np.argsort(-probs)[:top_k]
        top_probs = probs[top_indices] / (probs.sum() + 1e-9)
        
        return top_indices, top_probs
        
    def evaluate(self, X_val, y_val, batch_size=None, top_k=10):
        # Generate the exact same predictions for all instances
        probs = self.item_counts.copy()
        probs[0] = 0.0
        
        # Tile for all validations
        all_preds = np.tile(probs, (len(y_val), 1))
        
        return evaluate_predictions(all_preds, y_val, k=top_k)
        
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.item_counts, f)
            
    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.item_counts = pickle.load(f)
            return True
        return False
