import numpy as np
import pandas as pd

def preprocess_data(df, min_session_len=3, min_item_support=2):
    # Filter rare items
    item_counts = df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_item_support].index
    df = df[df['item_id'].isin(valid_items)].copy()
    
    # Filter short sessions
    session_counts = df['session_id'].value_counts()
    valid_sessions = session_counts[session_counts >= min_session_len].index
    df = df[df['session_id'].isin(valid_sessions)].copy()
    
    # Sort by timestamp
    df = df.sort_values(by=['session_id', 'timestamp'])
    
    # Encode item IDs (0 is reserved for padding)
    item2idx = {item: idx + 1 for idx, item in enumerate(df['item_id'].unique())}
    idx2item = {idx: item for item, idx in item2idx.items()}
    
    df['item_idx'] = df['item_id'].map(item2idx)
    
    return df, item2idx, idx2item

def generate_sequences(df, max_seq_len):
    sessions = df.groupby('session_id')['item_idx'].apply(list).to_dict()
    
    inputs = []
    targets = []
    
    for session_id, seq in sessions.items():
        if len(seq) < 2:
            continue
            
        for i in range(1, len(seq)):
            # Sequence truncation handling
            seq_in = seq[max(0, i - max_seq_len):i]
            target = seq[i]
            
            # Pad sequence if necessary
            pad_len = max_seq_len - len(seq_in)
            seq_in_padded = [0] * pad_len + seq_in
            
            inputs.append(seq_in_padded)
            targets.append(target)
            
    return np.array(inputs), np.array(targets)
