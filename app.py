import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, List, Tuple
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go

st.set_page_config(
    page_title="Token & Embedding Visualizer",
    layout="wide"
)

COLORS = {
    'Special': '#FFB6C1', 
    'Subword': '#98FB98', 
    'Word': '#87CEFA',    
    'Punctuation': '#DDA0DD'
}

@st.cache_resource
def load_models_and_tokenizers() -> Tuple[Dict, Dict]:
    """Load tokenizers and models with error handling"""
    model_names = {
        "BERT": "bert-base-uncased",
        "RoBERTa": "roberta-base",
        "DistilBERT": "distilbert-base-uncased",
        "MPNet": "microsoft/mpnet-base",
        "DeBERTa": "microsoft/deberta-base",
    }
    
    tokenizers = {}
    models = {}
    
    for name, model_name in model_names.items():
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(model_name)
            models[name] = AutoModel.from_pretrained(model_name)
            st.success(f"✓ Loaded {name}")
        except Exception as e:
            st.warning(f"× Failed to load {name}: {str(e)}")
    
    return tokenizers, models

def classify_token(token: str) -> str:
    if token.startswith(('##', '▁', 'Ġ', '_', '.')):
        return 'Subword'
    elif token in ['[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '[PAD]', '[MASK]', '<mask>']:
        return 'Special'
    elif token in [',', '.', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}']:
        return 'Punctuation'
    else:
        return 'Word'

@torch.no_grad()
def get_embeddings(text: str, model, tokenizer) -> Tuple[torch.Tensor, List[str]]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[0]  # Get first batch
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return embeddings, tokens

def visualize_embeddings(embeddings: torch.Tensor, tokens: List[str], method: str = 'PCA') -> go.Figure:
    embed_array = embeddings.numpy()
    
    if method == 'PCA':
        reducer = PCA(n_components=3)
        reduced_embeddings = reducer.fit_transform(embed_array)
        variance_explained = reducer.explained_variance_ratio_
        method_info = f"Total variance explained: {sum(variance_explained):.2%}"
    else:  # t-SNE
        reducer = TSNE(n_components=3, random_state=42, perplexity=min(30, len(tokens)-1))
        reduced_embeddings = reducer.fit_transform(embed_array)
        method_info = "t-SNE embedding (perplexity: {})".format(reducer.perplexity)
    
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'z': reduced_embeddings[:, 2],
        'token': tokens,
        'type': [classify_token(t) for t in tokens]
    })
    
    fig = go.Figure()
    
    for token_type in df['type'].unique():
        mask = df['type'] == token_type
        fig.add_trace(go.Scatter3d(
            x=df[mask]['x'],
            y=df[mask]['y'],
            z=df[mask]['z'],
            mode='markers+text',
            name=token_type,
            text=df[mask]['token'],
            hovertemplate="Token: %{text}<br>Type: " + token_type + "<extra></extra>",
            marker=dict(
                size=8,
                color=COLORS[token_type],
                opacity=0.8
            )
        ))
    
    fig.update_layout(
        title=f"{method} Visualization of Token Embeddings<br><sup>{method_info}</sup>",
        scene=dict(
            xaxis_title=f"{method}_1",
            yaxis_title=f"{method}_2",
            zaxis_title=f"{method}_3"
        ),
        width=800,
        height=800
    )
    
    return fig

def compute_token_similarities(embeddings: torch.Tensor, tokens: List[str]) -> pd.DataFrame:
    normalized_embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    sim_df = pd.DataFrame(similarities.numpy(), columns=tokens, index=tokens)
    return sim_df

st.title("🔤 Token & Embedding Visualizer")

# Load models and tokenizers
tokenizers, models = load_models_and_tokenizers()

token_tab, embedding_tab, similarity_tab = st.tabs([
    "Token Visualization",
    "Embedding Visualization",
    "Token Similarities"
])

default_text = "Hello world! Let's analyze how neural networks process language. The transformer architecture revolutionized NLP."
text_input = st.text_area("Enter text to analyze:", value=default_text, height=100)

with token_tab:
    st.markdown("""
    Token colors represent:
    - 🟦 Blue: Complete words
    - 🟩 Green: Subwords
    - 🟨 Pink: Special tokens
    - 🟪 Purple: Punctuation
    """)
    
    selected_models = st.multiselect(
        "Select models to compare tokens",
        options=list(tokenizers.keys()),
        default=["BERT", "RoBERTa"],
        max_selections=4
    )
    
    if text_input and selected_models:
        cols = st.columns(len(selected_models))
        
        for idx, model_name in enumerate(selected_models):
            with cols[idx]:
                st.subheader(model_name)
                tokenizer = tokenizers[model_name]
                
                tokens = tokenizer.tokenize(text_input)
                token_ids = tokenizer.encode(text_input)
                
                if len(tokens) != len(token_ids):
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                
                st.metric("Tokens", len(tokens))
                
                html_tokens = []
                for token in tokens:
                    color = COLORS[classify_token(token)]
                    token_text = token.replace('<', '&lt;').replace('>', '&gt;')
                    html_tokens.append(
                        f'<span style="background-color: {color}; padding: 2px 4px; '
                        f'margin: 2px; border-radius: 3px; font-family: monospace;">'
                        f'{token_text}</span>'
                    )
                
                st.markdown(
                    '<div style="background-color: white; padding: 10px; '
                    'border-radius: 5px; border: 1px solid #ddd;">'
                    f'{"".join(html_tokens)}</div>',
                    unsafe_allow_html=True
                )

with embedding_tab:
    st.markdown("""
    This tab shows how tokens are embedded in the model's vector space.
    - Compare different dimensionality reduction techniques
    - Observe clustering of similar tokens
    - Explore the relationship between different token types
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select model for embedding visualization",
            options=list(models.keys())
        )
    
    with col2:
        viz_method = st.radio(
            "Select visualization method",
            options=['PCA', 't-SNE'],
            horizontal=True
        )
    
    if text_input and selected_model:
        with st.spinner(f"Generating embeddings with {selected_model}..."):
            embeddings, tokens = get_embeddings(
                text_input,
                models[selected_model],
                tokenizers[selected_model]
            )
            
            fig = visualize_embeddings(embeddings, tokens, viz_method)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Embedding Statistics"):
                embed_stats = pd.DataFrame({
                    'Token': tokens,
                    'Type': [classify_token(t) for t in tokens],
                    'Mean': embeddings.mean(dim=1).numpy(),
                    'Std': embeddings.std(dim=1).numpy(),
                    'Norm': torch.norm(embeddings, dim=1).numpy()
                })
                st.dataframe(embed_stats, use_container_width=True)

with similarity_tab:
    st.markdown("""
    Explore token similarities based on their embedding representations.
    - Darker colors indicate higher similarity
    - Hover over cells to see exact similarity scores
    """)
    
    if text_input and selected_model:
        with st.spinner("Computing token similarities..."):
            sim_df = compute_token_similarities(embeddings, tokens)
            
            fig = px.imshow(
                sim_df,
                labels=dict(color="Cosine Similarity"),
                color_continuous_scale="RdYlBu",
                aspect="auto"
            )
            fig.update_layout(
                title="Token Similarity Matrix",
                width=800,
                height=800
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Most Similar Token Pairs")
            sim_matrix = sim_df.values
            np.fill_diagonal(sim_matrix, 0)  # Exclude self-similarities
            top_k = min(10, len(tokens))
            
            pairs = []
            for i in range(len(tokens)):
                for j in range(i+1, len(tokens)):
                    pairs.append((tokens[i], tokens[j], sim_matrix[i, j]))
            
            top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_k]
            
            for token1, token2, sim in top_pairs:
                st.write(f"'{token1}' — '{token2}': {sim:.3f}")

st.markdown("---")
st.markdown("""
    💡 **Tips:**
    - Try comparing how different models tokenize and embed the same text
    - Use PCA for global structure and t-SNE for local relationships
    - Check the similarity matrix for interesting token relationships
    - Experiment with different text types (technical, casual, mixed)
""")
