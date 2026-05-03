import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import os

from environments.synthetic import load_environment as load_synthetic
from environments.ecommerce import load_environment as load_ecommerce
from environments.movielens import load_environment as load_movielens
from utils.preprocessing import preprocess_data, generate_sequences

from agents.gru4rec import GRU4RecAgent
from agents.sasrec import SASRecAgent
from agents.popularity import PopularityAgent

st.set_page_config(page_title="SBRS Framework", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #1E2127; border-radius: 4px 4px 0px 0px; }
    .stTabs [aria-selected="true"] { background-color: #2E323A; border-bottom: 2px solid #FF4B4B; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Advanced Session-Based Recommendation System")
st.markdown("A unified framework supporting multiple environments and models for sequence-aware recommendations.")

# Session State Initialization
if 'df' not in st.session_state: st.session_state.df = None
if 'model_metrics' not in st.session_state: st.session_state.model_metrics = {}
if 'item2idx' not in st.session_state: st.session_state.item2idx = {}
if 'idx2item' not in st.session_state: st.session_state.idx2item = {}
if 'trained_agents' not in st.session_state: st.session_state.trained_agents = {}

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    env_name = st.selectbox("1. Select Environment", 
                            ["Synthetic Data", "E-Commerce Clickstream", "MovieLens Sessions", "Upload Custom Dataset"])
                            
    agent_name = st.selectbox("2. Select Agent (Model)", 
                              ["GRU4Rec", "SASRec", "Popularity Baseline"])
                              
    st.markdown("---")
    st.subheader("3. Hyperparameters")
    
    # Dynamically infer valid top_k based on dataset size
    if 'item2idx' in st.session_state and st.session_state.item2idx:
        max_k = min(20, len(st.session_state.item2idx))
    else:
        max_k = 20
    top_k = st.slider("Top-K Value (for Eval)", 1, max(1, max_k), min(10, max_k))
    
    if agent_name != "Popularity Baseline":
        lr = st.number_input("Learning Rate", value=0.001, format="%.4f")
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
        batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=2)
        seq_len = st.number_input("Sequence Length", min_value=5, max_value=50, value=20)
        emb_dim = st.selectbox("Embedding Dimension", [32, 64, 128], index=1)
        dropout = st.slider("Dropout Rate", 0.0, 0.9, 0.2)
        
        if agent_name == "GRU4Rec":
            hidden_size = st.selectbox("Hidden Size", [64, 128, 256], index=1)
        elif agent_name == "SASRec":
            num_heads = st.selectbox("Attention Heads", [1, 2, 4, 8], index=1)
    else:
        # Dummy values for popularity baseline
        epochs, batch_size, seq_len = 1, 256, 20
        
    seed = st.number_input("Random Seed", value=42)
    
    start_train = st.button("🚀 Train Agent", use_container_width=True)

# Load Environment Data
if env_name == "Synthetic Data":
    st.session_state.df, env_desc = load_synthetic()
elif env_name == "E-Commerce Clickstream":
    st.session_state.df, env_desc = load_ecommerce()
elif env_name == "MovieLens Sessions":
    st.session_state.df, env_desc = load_movielens()
elif env_name == "Upload Custom Dataset":
    env_desc = "Custom Uploaded Dataset"
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.sidebar.markdown("**Map your columns:**")
            col_sess = st.sidebar.selectbox("Session ID Column", raw_df.columns, index=0)
            col_item = st.sidebar.selectbox("Item ID Column", raw_df.columns, index=1 if len(raw_df.columns)>1 else 0)
            col_time = st.sidebar.selectbox("Timestamp Column (Optional)", ["None"] + list(raw_df.columns))
            
            if st.sidebar.button("Process Uploaded Data"):
                mapped_df = raw_df.rename(columns={col_sess: 'session_id', col_item: 'item_id'})
                if col_time != "None":
                    mapped_df = mapped_df.rename(columns={col_time: 'timestamp'})
                else:
                    # Add dummy timestamp if none provided to satisfy preprocessing sorting
                    mapped_df['timestamp'] = np.arange(len(mapped_df))
                
                # Keep only required columns
                st.session_state.df = mapped_df[['session_id', 'item_id', 'timestamp']]
                st.sidebar.success("Dataset mapped and loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

# --- MAIN UI TABS ---
tab_data, tab_train, tab_eval, tab_recs, tab_compare = st.tabs(["🗂️ Data Explorer", "🚀 Training", "📊 Evaluation", "🔮 Recommendations", "📈 Comparison"])

with tab_data:
    st.header(f"Dataset Overview: {env_desc}")
    if st.session_state.df is not None:
        st.write("### Raw Data Sample")
        st.dataframe(st.session_state.df.head(100), use_container_width=True)
        
        st.write("### Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Interactions", len(st.session_state.df))
        col2.metric("Unique Sessions", st.session_state.df['session_id'].nunique())
        col3.metric("Unique Items", st.session_state.df['item_id'].nunique())
        
        st.write("### Session Length Distribution")
        session_lengths = st.session_state.df.groupby('session_id').size()
        fig = go.Figure(data=[go.Histogram(x=session_lengths, nbinsx=50)])
        fig.update_layout(title="Session Length Distribution", xaxis_title="Session Length", yaxis_title="Count", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data loaded. Please select an environment from the sidebar.")

with tab_train:
    st.header(f"Training Console: {agent_name} on {env_desc}")
    
    if start_train:
        # Set reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        with st.spinner("Preprocessing environment data..."):
            df_clean, item2idx, idx2item = preprocess_data(st.session_state.df)
            st.session_state.item2idx = item2idx
            st.session_state.idx2item = idx2item
            
            num_items = len(item2idx) + 1
            X, y = generate_sequences(df_clean, max_seq_len=seq_len)
            
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]
            
        st.info(f"Loaded {env_desc} | Items: {num_items} | Train seqs: {len(X_train)} | Val seqs: {len(X_val)}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.write(f"Compute Device: `{device}`")
        
        # Initialize Agent
        if agent_name == "GRU4Rec":
            agent = GRU4RecAgent(num_items, max_seq_len=seq_len, embedding_dim=emb_dim, 
                                 hidden_size=hidden_size, dropout=dropout, lr=lr, device=device)
        elif agent_name == "SASRec":
            agent = SASRecAgent(num_items, max_seq_len=seq_len, embedding_dim=emb_dim, 
                                num_heads=num_heads, dropout=dropout, lr=lr, device=device)
        else:
            agent = PopularityAgent(num_items, max_seq_len=seq_len)
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def ui_callback(epoch, total_epochs, train_loss, val_metrics):
            progress_bar.progress(epoch / total_epochs)
            status_text.text(f"Epoch {epoch}/{total_epochs} | Loss: {train_loss:.4f} | HR@{top_k}: {val_metrics.get(f'HR@{top_k}', 0):.4f}")
            
        with st.spinner(f"Training {agent_name}..."):
            history = agent.train(X_train, y_train, X_val, y_val, epochs=epochs, 
                                  batch_size=batch_size, top_k=top_k, ui_callback=ui_callback)
            
        st.success(f"{agent_name} trained successfully!")
        
        agent_id = f"{agent_name}_{env_name.replace(' ', '')}"
        
        # Save model
        agent.save(f"models/{agent_id}.pt")
        
        # Store in session state
        st.session_state.trained_agents[agent_id] = agent
        st.session_state.model_metrics[agent_id] = history
        st.session_state.seq_len = seq_len
        
        # Plot Loss Curve
        if agent_name != "Popularity Baseline":
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['train_loss'], mode='lines+markers', name='Train Loss'))
            fig.update_layout(title="Training Loss over Epochs", template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

with tab_eval:
    st.header("Agent Evaluation")
    if not st.session_state.model_metrics:
        st.warning("Train an agent first.")
    else:
        selected_agent = st.selectbox("Select Trained Agent to View", list(st.session_state.model_metrics.keys()))
        history = st.session_state.model_metrics[selected_agent]
        val_metrics = history['val_metrics']
        
        final_metrics = val_metrics[-1]
        
        st.subheader("Final Evaluation Metrics")
        try:
            m_cols = st.columns(6)
            m_cols[0].metric(f"HR@{top_k}", f"{final_metrics.get(f'HR@{top_k}', 0):.4f}")
            m_cols[1].metric(f"MRR@{top_k}", f"{final_metrics.get(f'MRR@{top_k}', 0):.4f}")
            m_cols[2].metric(f"Precision@{top_k}", f"{final_metrics.get(f'Precision@{top_k}', 0):.4f}")
            m_cols[3].metric(f"Recall@{top_k}", f"{final_metrics.get(f'Recall@{top_k}', 0):.4f}")
            m_cols[4].metric("Coverage", f"{final_metrics.get('Coverage', 0):.4f}")
            m_cols[5].metric("Novelty", f"{final_metrics.get('Novelty', 0):.4f}")
        except Exception as e:
            st.error(f"Error displaying metrics: {e}. Available metrics: {list(final_metrics.keys())}")
            
        if f'HR@{top_k}' not in final_metrics:
            st.warning(f"Metrics for Top-K={top_k} were not computed during training. Shown values are 0. Please train again with Top-K={top_k} or adjust the slider. Available keys: {list(final_metrics.keys())}")
        
        if len(val_metrics) > 1:
            try:
                epochs_arr = list(range(1, len(val_metrics) + 1))
                fig = make_subplots(rows=1, cols=2, subplot_titles=(f"HR@{top_k}", f"MRR@{top_k}"))
                fig.add_trace(go.Scatter(x=epochs_arr, y=[m.get(f'HR@{top_k}', 0) for m in val_metrics], name=f"HR@{top_k}"), row=1, col=1)
                fig.add_trace(go.Scatter(x=epochs_arr, y=[m.get(f'MRR@{top_k}', 0) for m in val_metrics], name=f"MRR@{top_k}"), row=1, col=2)
                fig.update_layout(height=400, title_text="Evaluation Progress over Epochs", template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering chart: {e}")

with tab_recs:
    st.header("🔮 Top-K Inference Engine")
    if not st.session_state.trained_agents:
        st.warning("Train an agent first.")
    else:
        infer_agent_id = st.selectbox("Select Agent for Inference", list(st.session_state.trained_agents.keys()))
        agent = st.session_state.trained_agents[infer_agent_id]
        
        sample_items = list(st.session_state.item2idx.keys())[:15]
        st.caption(f"Valid Item IDs: {', '.join(map(str, sample_items))} ...")
        
        user_input = st.text_input("Enter User Interaction Sequence (comma-separated)", "1, 2, 3")
        k_recs = st.slider("Number of Recommendations", 1, 20, 10)
        
        if st.button("Generate Recommendations"):
            try:
                input_items = [int(x.strip()) for x in user_input.split(',')]
                item2idx = st.session_state.item2idx
                idx2item = st.session_state.idx2item
                
                # Cold-start fallback strategy
                seq_indices = [item2idx[i] for i in input_items if i in item2idx]
                
                if not seq_indices:
                    st.warning("All items are new (Cold-Start). Falling back to Popularity Baseline...")
                    # Since we don't store popularity globally, we can just return random/empty or handle properly
                    st.error("Cold start detected! For a true popularity fallback, train the Popularity Baseline.")
                else:
                    seq_len_req = st.session_state.seq_len
                    if len(seq_indices) > seq_len_req:
                        seq_indices = seq_indices[-seq_len_req:]
                    else:
                        seq_indices = [0] * (seq_len_req - len(seq_indices)) + seq_indices
                        
                    top_indices, top_probs = agent.predict(seq_indices, top_k=k_recs)
                    
                    recs = [{"Rank": i+1, "Item ID": idx2item.get(idx, 'Unknown'), "Confidence": f"{p*100:.2f}%"} 
                            for i, (idx, p) in enumerate(zip(top_indices, top_probs))]
                            
                    st.table(pd.DataFrame(recs).set_index("Rank"))
                    
            except Exception as e:
                st.error(f"Error processing input: {e}")

with tab_compare:
    st.header("📈 Multi-Agent Comparison")
    if len(st.session_state.trained_agents) < 2:
        st.info("Train at least 2 different agents to view comparison.")
    else:
        comp_data = []
        for a_id, history in st.session_state.model_metrics.items():
            fm = history['val_metrics'][-1]
            comp_data.append({
                "Agent (Environment)": a_id,
                f"HR@{top_k}": fm.get(f'HR@{top_k}', 0),
                f"MRR@{top_k}": fm.get(f'MRR@{top_k}', 0),
                f"NDCG@{top_k}": fm.get(f'NDCG@{top_k}', 0),
                "Coverage": fm.get('Coverage', 0),
                "Novelty": fm.get('Novelty', 0)
            })
            
        df_comp = pd.DataFrame(comp_data).set_index("Agent (Environment)")
        st.dataframe(df_comp.style.format("{:.4f}"))
        
        # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_comp.index, y=df_comp[f'HR@{top_k}'], name=f'HR@{top_k}'))
        fig.add_trace(go.Bar(x=df_comp.index, y=df_comp[f'MRR@{top_k}'], name=f'MRR@{top_k}'))
        fig.update_layout(title="Metric Comparison", template='plotly_dark', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Export
        csv = df_comp.to_csv()
        st.download_button("📥 Export Results to CSV", data=csv, file_name="sbrs_comparison.csv", mime="text/csv")
