import numpy as np

def hit_rate(rank_list, ground_truth):
    return 1.0 if ground_truth in rank_list else 0.0

def mrr(rank_list, ground_truth):
    if ground_truth in rank_list:
        return 1.0 / (rank_list.index(ground_truth) + 1)
    return 0.0

def precision_at_k(rank_list, ground_truth):
    return 1.0 / len(rank_list) if ground_truth in rank_list else 0.0

def recall_at_k(rank_list, ground_truth):
    return 1.0 if ground_truth in rank_list else 0.0

def ndcg_at_k(rank_list, ground_truth):
    if ground_truth in rank_list:
        return 1.0 / np.log2(rank_list.index(ground_truth) + 2)
    return 0.0

def evaluate_predictions(predictions, targets, k=10):
    """
    predictions: (batch_size, num_items)
    targets: (batch_size,)
    """
    top_k_items = np.argsort(-predictions, axis=1)[:, :k]
    
    hr, mrr_score, prec, rec, ndcg = [], [], [], [], []
    recommended_items = set()
    
    for i in range(len(targets)):
        pred_list = list(top_k_items[i])
        truth = targets[i]
        
        hr.append(hit_rate(pred_list, truth))
        mrr_score.append(mrr(pred_list, truth))
        prec.append(precision_at_k(pred_list, truth))
        rec.append(recall_at_k(pred_list, truth))
        ndcg.append(ndcg_at_k(pred_list, truth))
        
        recommended_items.update(pred_list)
        
    num_items = predictions.shape[1]
    # Coverage: unique recommended items / total items
    coverage = len(recommended_items) / float(num_items - 1) if num_items > 1 else 0
    
    # Novelty: inverse popularity approximation
    # For a real implementation, we would pass in training frequencies
    novelty = 1.0 - (len(recommended_items) / (len(targets) * k)) # Placeholder approximation
    
    return {
        f'HR@{k}': np.mean(hr),
        f'MRR@{k}': np.mean(mrr_score),
        f'Precision@{k}': np.mean(prec),
        f'Recall@{k}': np.mean(rec),
        f'NDCG@{k}': np.mean(ndcg),
        f'Coverage': coverage,
        f'Novelty': max(0.0, novelty)
    }
