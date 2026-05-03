# Session-Based Recommendation System using Deep Learning

A production-grade framework for training, evaluating, and comparing Session-Based Recommendation models using an RL-like abstraction of Environments and Agents. This project predicts a user's next action based on their current sequence of interactions.

## 🌟 Architecture (How it Works)

The codebase is highly modular and broken down into three main components:

1. **Environments (`/environments`)**: These act as data simulators. They take raw data and convert it into sequences of user actions (states).
   - **E-Commerce Clickstream**: Real-world shopping data.
   - **MovieLens Sessions**: Movie watching/rating sequences.
   - **Synthetic Data**: Randomly generated data for quick testing.
   - **Custom Uploads**: Supports uploading custom datasets via CSV.

2. **Agents (`/agents`)**: These are the AI models that learn from the environments.
   - **Popularity Baseline**: A simple model that always recommends the most popular items. Used as a baseline for comparison.
   - **GRU4Rec**: A powerful Recurrent Neural Network (RNN) designed specifically for session-based recommendations. It understands the "flow" of a user's session.
   - **SASRec**: A state-of-the-art Transformer-based model (Self-Attention) that looks at the entire session context to weigh which previous items are most important for the next prediction.

3. **Training & Metrics (`/training` & `/utils`)**: Handles PyTorch training loops, sequence masking, padding, and early stopping. Evaluates models using industry-standard metrics:
   - **HR@K** (Hit Rate)
   - **MRR** (Mean Reciprocal Rank)
   - **NDCG** (Normalized Discounted Cumulative Gain)
   - **Coverage & Novelty**

## 🚀 Quickstart (Local Deployment)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 💻 How to Use the Streamlit Dashboard

Once the app is running, navigate the dashboard using the following workflow:

### Step 1: Configure Your Environment (Left Sidebar)
Start by setting up *what* data the model will learn from.
- Select your **Environment** (e.g., Synthetic Data for quick testing).
- Adjust the number of users/items.
- Click **"Initialize Environment"** to process the data and prepare training sets.

### Step 2: Configure Your Agent (Left Sidebar)
Choose the AI model you want to train.
- Select your **Agent** (e.g., `SASRec` or `GRU4Rec`).
- Adjust Hyperparameters like Learning Rate, Embedding Dimension, and Batch Size.
- Click **"Initialize Agent"**.

### Step 3: Train the Model (Main Screen - Training Tab)
- Go to the **"Training & Metrics"** tab.
- Select the number of **Epochs**.
- Click **"Start Training"**. Watch the live progress bar and real-time loss chart. Once finished, a scorecard with final metrics (HR@10, MRR@10) will appear.

### Step 4: Test it Yourself (Interactive Evaluation Tab)
Test the trained model in real-time.
- Go to the **"Interactive Evaluation"** tab.
- **User Interaction Sequence:** This represents your current session history. Input a sequence of Item IDs (e.g., `2, 15, 8` or select from a dropdown) to pretend you are a user navigating the site.
- Click **"Get Recommendations"**. The AI will instantly calculate and display a bar chart of the Top-K items it predicts you will click on next based on your sequence.

### Step 5: Compare Models
Train multiple models consecutively (e.g., train the Popularity Baseline, then train SASRec). The **Model Comparison** tab will automatically plot a radar chart comparing their performance metrics side-by-side!

---
*Developed for research and application of Sequence Modeling in Recommender Systems.*
