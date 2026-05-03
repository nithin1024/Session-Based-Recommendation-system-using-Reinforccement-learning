# Session-Based Recommendation System using Reinforcement Learning

A complete framework for training, evaluating, and comparing Session-Based Recommendation models using an RL-like abstraction of Environments and Agents.

## 🌟 Architecture

- **Environments**: E-commerce, MovieLens, and Synthetic datasets that expose a unified session interface.
- **Agents**: GRU4Rec, SASRec, and Popularity Baselines implemented modularly.
- **Training System**: Sequence masking, padding, early stopping, and PyTorch training loops.
- **Evaluation**: HR@K, MRR, Precision, Recall, NDCG, Coverage, and Novelty.

## 🚀 Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```
