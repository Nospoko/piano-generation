import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Tuple


def create_loss_analysis_plots(df: pd.DataFrame, 
                             model_size: float,
                             training_type: str,
                             metrics: List[str]) -> Tuple[go.Figure, go.Figure]:
    """Create plots comparing validation and training loss with metrics"""
    # Filter data
    mask = (df['model_size_M'] == model_size) & (df['training_type'] == training_type)
    filtered_df = df[mask].sort_values('total_tokens_M')
    
    if filtered_df.empty:
        return None, None
    
    # Create validation loss plot
    val_loss_fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=[f"{m.replace('_mean', '').replace('_', ' ').title()} vs Validation Loss"
                       for m in metrics],
        x_title="Validation Loss"
    )
    
    # Create training loss plot
    train_loss_fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=[f"{m.replace('_mean', '').replace('_', ' ').title()} vs Training Loss"
                       for m in metrics],
        x_title="Training Loss"
    )
    
    for i, metric in enumerate(metrics, 1):
        # Validation loss scatter plot
        hover_text = [
            f"Model: {model}<br>"
            f"Size: {size}M<br>"
            f"Tokens: {tokens:.1f}M<br>"
            f"Val Loss: {val_loss:.3f}<br>"
            f"{metric.replace('_mean', '')}: {value:.3f}"
            for model, size, tokens, val_loss, value in 
            zip(filtered_df['model_name'], 
                filtered_df['model_size_M'], 
                filtered_df['total_tokens_M'],
                filtered_df['best_val_loss'],
                filtered_df[metric])
        ]
        
        val_loss_fig.add_trace(
            go.Scatter(x=filtered_df['best_val_loss'],
                      y=filtered_df[metric],
                      mode='markers',
                      hovertext=hover_text,
                      hoverinfo='text',
                      marker=dict(
                          size=10,
                          color=filtered_df['total_tokens_M'],
                          colorscale='Viridis',
                          showscale=True,
                          colorbar=dict(title="Tokens (M)")
                      )),
            row=i, col=1
        )
        
        # Add trendline for validation loss
        if len(filtered_df) > 2:
            z = np.polyfit(filtered_df['best_val_loss'], filtered_df[metric], 1)
            p = np.poly1d(z)
            x_range = np.linspace(filtered_df['best_val_loss'].min(), 
                                filtered_df['best_val_loss'].max(), 
                                100)
            val_loss_fig.add_trace(
                go.Scatter(x=x_range,
                          y=p(x_range),
                          mode='lines',
                          line=dict(dash='dash', color='red'),
                          showlegend=False),
                row=i, col=1
            )
        
        # Training loss scatter plot
        hover_text = [
            f"Model: {model}<br>"
            f"Size: {size}M<br>"
            f"Tokens: {tokens:.1f}M<br>"
            f"Train Loss: {train_loss:.3f}<br>"
            f"{metric.replace('_mean', '')}: {value:.3f}"
            for model, size, tokens, train_loss, value in 
            zip(filtered_df['model_name'], 
                filtered_df['model_size_M'], 
                filtered_df['total_tokens_M'],
                filtered_df['train_loss'],
                filtered_df[metric])
        ]
        
        train_loss_fig.add_trace(
            go.Scatter(x=filtered_df['train_loss'],
                      y=filtered_df[metric],
                      mode='markers',
                      hovertext=hover_text,
                      hoverinfo='text',
                      marker=dict(
                          size=10,
                          color=filtered_df['total_tokens_M'],
                          colorscale='Viridis',
                          showscale=True,
                          colorbar=dict(title="Tokens (M)")
                      )),
            row=i, col=1
        )
        
        # Add trendline for training loss
        if len(filtered_df) > 2:
            z = np.polyfit(filtered_df['train_loss'], filtered_df[metric], 1)
            p = np.poly1d(z)
            x_range = np.linspace(filtered_df['train_loss'].min(), 
                                filtered_df['train_loss'].max(), 
                                100)
            train_loss_fig.add_trace(
                go.Scatter(x=x_range,
                          y=p(x_range),
                          mode='lines',
                          line=dict(dash='dash', color='red'),
                          showlegend=False),
                row=i, col=1
            )
    
    # Update layouts
    val_loss_fig.update_layout(
        height=300*len(metrics),
        title=f'Metrics vs Validation Loss for {model_size}M Model ({training_type})',
        showlegend=False
    )
    
    train_loss_fig.update_layout(
        height=300*len(metrics),
        title=f'Metrics vs Training Loss for {model_size}M Model ({training_type})',
        showlegend=False
    )
    
    return val_loss_fig, train_loss_fig

def create_loss_progression_plot(df: pd.DataFrame, 
                               model_size: float,
                               training_type: str) -> go.Figure:
    """Create plot showing loss progression during training"""
    # Filter data
    mask = (df['model_size_M'] == model_size) & (df['training_type'] == training_type)
    filtered_df = df[mask].sort_values('total_tokens_M')
    
    if filtered_df.empty:
        return None
    
    fig = go.Figure()
    
    # Add validation loss
    hover_text = [
        f"Model: {model}<br>"
        f"Tokens: {tokens:.1f}M<br>"
        f"Val Loss: {val_loss:.3f}"
        for model, tokens, val_loss in 
        zip(filtered_df['model_name'], 
            filtered_df['total_tokens_M'],
            filtered_df['best_val_loss'])
    ]
    
    fig.add_trace(
        go.Scatter(x=filtered_df['total_tokens_M'],
                  y=filtered_df['best_val_loss'],
                  mode='markers+lines',
                  name='Validation Loss',
                  hovertext=hover_text,
                  hoverinfo='text',
                  line=dict(color='blue'))
    )
    
    # Add training loss
    hover_text = [
        f"Model: {model}<br>"
        f"Tokens: {tokens:.1f}M<br>"
        f"Train Loss: {train_loss:.3f}"
        for model, tokens, train_loss in 
        zip(filtered_df['model_name'], 
            filtered_df['total_tokens_M'],
            filtered_df['train_loss'])
    ]
    
    fig.add_trace(
        go.Scatter(x=filtered_df['total_tokens_M'],
                  y=filtered_df['train_loss'],
                  mode='markers+lines',
                  name='Training Loss',
                  hovertext=hover_text,
                  hoverinfo='text',
                  line=dict(color='red'))
    )
    
    fig.update_layout(
        title=f'Loss Progression for {model_size}M Model ({training_type})',
        xaxis_title="Training Tokens (M)",
        yaxis_title="Loss",
        showlegend=True
    )
    
    return fig


def extract_training_type(model_name: str) -> str:
    """Extract training type from model name"""
    if 'subsequence' in model_name.lower():
        return 'subsequence'
    elif 'pretraining' in model_name.lower():
        return 'pretraining'
    else:
        return 'other'

def load_and_process_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the metrics data"""
    df = pd.read_csv(file_path)
    
    # Convert tokens to millions for better readability
    df['total_tokens_M'] = df['total_tokens'] / 1_000_000
    
    # Extract training type
    df['training_type'] = df['model_name'].apply(extract_training_type)
    
    # Create token ranges for grouping
    df['token_range'] = pd.qcut(df['total_tokens_M'], 
                               q=5, 
                               labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
    
    # Sort by model size and tokens
    df = df.sort_values(['model_size_M', 'total_tokens_M'])
    
    return df

def plot_metrics_by_size(df: pd.DataFrame, 
                        token_range: str,
                        training_type: str,
                        metrics: List[str]) -> go.Figure:
    """Create scatter plots of metrics vs model size for specific token range"""
    # Filter data
    mask = (df['token_range'] == token_range) & (df['training_type'] == training_type)
    filtered_df = df[mask]
    
    if filtered_df.empty:
        return None
    
    fig = make_subplots(rows=len(metrics), cols=1,
                        subplot_titles=[f"{m.replace('_mean', '').replace('_', ' ').title()} vs Model Size"
                                      for m in metrics])
    
    for i, metric in enumerate(metrics, 1):
        # Add scatter plot
        hover_text = [
            f"Model: {model}<br>"
            f"Size: {size}M<br>"
            f"Tokens: {tokens:.1f}M<br>"
            f"{metric.replace('_mean', '')}: {value:.3f}"
            for model, size, tokens, value in 
            zip(filtered_df['model_name'], 
                filtered_df['model_size_M'], 
                filtered_df['total_tokens_M'],
                filtered_df[metric])
        ]
        
        fig.add_trace(
            go.Scatter(x=filtered_df['model_size_M'],
                      y=filtered_df[metric],
                      mode='markers+text',
                      text=filtered_df['model_size_M'].astype(str) + 'M',
                      textposition='top center',
                      hovertext=hover_text,
                      hoverinfo='text',
                      name=metric.replace('_mean', '').replace('_', ' ').title(),
                      marker=dict(size=10)),
            row=i, col=1
        )
        
        # Add trendline if we have enough points
        if len(filtered_df) > 2:
            z = np.polyfit(filtered_df['model_size_M'], filtered_df[metric], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(x=filtered_df['model_size_M'],
                          y=p(filtered_df['model_size_M']),
                          mode='lines',
                          name=f'Trend {metric}',
                          line=dict(dash='dash')),
                row=i, col=1
            )
    
    fig.update_layout(
        height=300*len(metrics),
        title=f'Metrics vs Model Size (Token Range: {token_range}, Type: {training_type})',
        showlegend=False
    )
    
    return fig

def create_token_progression_plot(df: pd.DataFrame, 
                                model_size: float,
                                training_type: str,
                                metrics: List[str]) -> go.Figure:
    """Create plots showing metric progression with training tokens for specific model size"""
    # Filter data
    mask = (df['model_size_M'] == model_size) & (df['training_type'] == training_type)
    filtered_df = df[mask].sort_values('total_tokens_M')
    
    if filtered_df.empty:
        return None
    
    fig = make_subplots(rows=len(metrics), cols=1,
                        subplot_titles=[f"{m.replace('_mean', '').replace('_', ' ').title()} vs Training Tokens"
                                      for m in metrics])
    
    for i, metric in enumerate(metrics, 1):
        hover_text = [
            f"Model: {model}<br>"
            f"Size: {size}M<br>"
            f"Tokens: {tokens:.1f}M<br>"
            f"{metric.replace('_mean', '')}: {value:.3f}"
            for model, size, tokens, value in 
            zip(filtered_df['model_name'], 
                filtered_df['model_size_M'], 
                filtered_df['total_tokens_M'],
                filtered_df[metric])
        ]
        
        fig.add_trace(
            go.Scatter(x=filtered_df['total_tokens_M'],
                      y=filtered_df[metric],
                      mode='markers+lines',
                      hovertext=hover_text,
                      hoverinfo='text',
                      name=metric),
            row=i, col=1
        )
    
    fig.update_layout(
        height=300*len(metrics),
        title=f'Training Progression for {model_size}M Model ({training_type})',
        showlegend=False
    )
    
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Model Metrics Analysis")
    
    st.title("MIDI Model Metrics Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Choose a metrics CSV file", type="csv")
    
    if uploaded_file is not None:
        df = load_and_process_data(uploaded_file)
        
        # Sidebar filters
        st.sidebar.header("Analysis Filters")
        
        training_type = st.sidebar.selectbox(
            "Select Training Type",
            options=sorted(df['training_type'].unique()),
            help="Filter models by training approach"
        )
        
        token_range = st.sidebar.selectbox(
            "Select Token Range",
            options=sorted(df['token_range'].unique()),
            help="Compare models trained with similar number of tokens"
        )
        
        metrics = st.sidebar.multiselect(
            "Select Metrics to Compare",
            options=['key_correlation_mean', 'pitch_correlation_mean', 'f1_score_mean'],
            default=['key_correlation_mean', 'pitch_correlation_mean', 'f1_score_mean'],
            help="Choose metrics to display in plots"
        )
        
        # Display basic statistics for filtered data
        filtered_df = df[
            (df['token_range'] == token_range) & 
            (df['training_type'] == training_type)
        ]
        
        st.header("Filtered Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Number of Models", len(filtered_df))
        with col2:
            st.metric("Token Range (M)", 
                     f"{filtered_df['total_tokens_M'].min():.1f} - {filtered_df['total_tokens_M'].max():.1f}")
        with col3:
            st.metric("Size Range (M)", 
                     f"{filtered_df['model_size_M'].min():.0f} - {filtered_df['model_size_M'].max():.0f}")
        with col4:
            st.metric("Avg Val Loss", 
                     f"{filtered_df['best_val_loss'].mean():.3f}")
        
        # Model Size Analysis
        st.header("Model Size Comparison")
        st.caption("Comparing models with similar training tokens")
        size_fig = plot_metrics_by_size(df, token_range, training_type, metrics)
        if size_fig:
            st.plotly_chart(size_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
        
        # Training Progression Analysis
        st.header("Training Progression Analysis")
        model_size = st.selectbox(
            "Select Model Size (M)",
            options=sorted(filtered_df['model_size_M'].unique())
        )
        
        progression_fig = create_token_progression_plot(df, model_size, training_type, metrics)
        if progression_fig:
            st.plotly_chart(progression_fig, use_container_width=True)
        else:
            st.warning("No training progression data available for selected model size")
        

        st.header("Loss Analysis")
        
        # Loss progression plot
        st.subheader("Loss Progression During Training")
        loss_prog_fig = create_loss_progression_plot(df, model_size, training_type)
        if loss_prog_fig:
            st.plotly_chart(loss_prog_fig, use_container_width=True)
        else:
            st.warning("No loss progression data available for selected model size")
        
        # Validation and Training Loss vs Metrics
        st.subheader("Loss vs Metrics Analysis")
        col1, col2 = st.columns(2)
        
        val_loss_fig, train_loss_fig = create_loss_analysis_plots(df, model_size, training_type, metrics)
        
        if val_loss_fig and train_loss_fig:
            with col1:
                st.plotly_chart(val_loss_fig, use_container_width=True)
            with col2:
                st.plotly_chart(train_loss_fig, use_container_width=True)
        else:
            st.warning("No loss analysis data available for selected model size")
        
        # Add loss statistics
        st.subheader("Loss Statistics")
        loss_stats = filtered_df.agg({
            'best_val_loss': ['mean', 'std', 'min', 'max'],
            'train_loss': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Validation Loss Stats:")
            st.dataframe(loss_stats['best_val_loss'])
        
        with col2:
            st.write("Training Loss Stats:")
            st.dataframe(loss_stats['train_loss'])
        
        # Calculate loss-metric correlations
        st.subheader("Loss-Metric Correlations")
        corr_data = []
        for metric in metrics:
            val_corr = filtered_df['best_val_loss'].corr(filtered_df[metric])
            train_corr = filtered_df['train_loss'].corr(filtered_df[metric])
            corr_data.append({
                'Metric': metric.replace('_mean', '').replace('_', ' ').title(),
                'Validation Loss Correlation': round(val_corr, 3),
                'Training Loss Correlation': round(train_corr, 3)
            })
        
        st.dataframe(pd.DataFrame(corr_data))

        
        # Detailed Data View
        st.header("Detailed Data View")
        st.dataframe(
            filtered_df[['model_name', 'model_size_M', 'total_tokens_M', 'best_val_loss'] + metrics]
            .sort_values(['model_size_M', 'total_tokens_M'])
        )
        
        # Download filtered data
        st.download_button(
            label="Download filtered data as CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name=f'filtered_metrics_{training_type}_{token_range}.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()