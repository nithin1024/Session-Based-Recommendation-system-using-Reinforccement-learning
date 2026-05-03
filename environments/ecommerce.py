import pandas as pd
import numpy as np
import os

def load_environment(num_sessions=1000):
    """
    Simulates a retail E-Commerce clickstream dataset.
    Longer sessions, highly skewed popularity (bestsellers).
    """
    np.random.seed(101)
    data = []
    timestamp = 1620000000
    num_items = 1000
    
    for session_id in range(1, num_sessions + 1):
        session_len = np.random.randint(2, 30)
        # Highly skewed item probabilities to mimic retail blockbusters
        item_probs = np.exp(-np.arange(1, num_items + 1) / 20)
        item_probs /= item_probs.sum()
        
        items = np.random.choice(np.arange(1, num_items + 1), size=session_len, p=item_probs)
        
        for item in items:
            data.append([session_id, item, timestamp])
            timestamp += np.random.randint(5, 60) # Fast clicks
            
    df = pd.DataFrame(data, columns=['session_id', 'item_id', 'timestamp'])
    return df, "E-Commerce Environment"
