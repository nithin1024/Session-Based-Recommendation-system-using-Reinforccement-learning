import pandas as pd
import numpy as np

def load_environment(num_sessions=1000):
    """
    Simulates a MovieLens-style dataset.
    Shorter sessions, moderate popularity skew, more unique items.
    """
    np.random.seed(202)
    data = []
    timestamp = 1640000000
    num_items = 2000
    
    for session_id in range(1, num_sessions + 1):
        session_len = np.random.randint(5, 15)
        # Moderate skew
        item_probs = np.exp(-np.arange(1, num_items + 1) / 100)
        item_probs /= item_probs.sum()
        
        items = np.random.choice(np.arange(1, num_items + 1), size=session_len, p=item_probs)
        
        for item in items:
            data.append([session_id, item, timestamp])
            timestamp += np.random.randint(3600, 86400) # Slower interactions (movies)
            
    df = pd.DataFrame(data, columns=['session_id', 'item_id', 'timestamp'])
    return df, "MovieLens Environment"
