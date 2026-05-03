import pandas as pd
import numpy as np

def load_environment(num_sessions=1000):
    """
    Generates a generic synthetic session dataset.
    """
    np.random.seed(42)
    data = []
    timestamp = 1600000000
    num_items = 500
    
    for session_id in range(1, num_sessions + 1):
        session_len = np.random.randint(3, 20)
        item_probs = np.exp(-np.arange(1, num_items + 1) / 50)
        item_probs /= item_probs.sum()
        
        items = np.random.choice(np.arange(1, num_items + 1), size=session_len, p=item_probs)
        
        for item in items:
            data.append([session_id, item, timestamp])
            timestamp += np.random.randint(10, 100)
            
    df = pd.DataFrame(data, columns=['session_id', 'item_id', 'timestamp'])
    return df, "Synthetic Environment"
