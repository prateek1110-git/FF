import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import numpy as np
import math
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Page Config
st.set_page_config(
    layout="wide",
    page_title="📊 Economic Indicator Analyzer",
    page_icon="📈"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .st-b7 {
        color: #2c3e50;
    }
    .css-18e3th9 {
        padding: 2rem 5rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .st-eb {
        background-color: #3498db !important;
        color: white !important;
    }
    .tab-content {
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("📊 Economic Indicator Analyzer")
st.markdown("""
Explore relationships between economic indicators through correlation, seasonality, and lead-lag analysis.
Upload your CSV files below to get started.
""")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Settings")
    analysis_type = st.radio(
        "Analysis Type",
        ["📈 Overview", "🔗 Correlation", "🔄 Seasonality", "⏩ Lead-Lag"],
        index=0
    )
    
    max_lag = st.slider(
        "Maximum Lag Period (months)",
        min_value=1,
        max_value=24,
        value=12,
        help="Set the maximum time lag to analyze"
    )
    
    st.markdown("---")
    st.markdown("**💡 Tips**")
    st.markdown("""
    - Upload CSV files with time series data
    - Ensure columns include dates and values
    - Larger datasets may take longer to process
    """)

# File Upload Section
uploaded_files = st.file_uploader(
    "📤 Upload CSV Files", 
    type=['csv'], 
    accept_multiple_files=True,
    help="Upload one or more CSV files containing economic indicator data"
)

# --- Core Functions ---
def preprocess_df(df, name=""):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # Try to automatically detect date and value columns
    date_cols = [c for c in df.columns if 'date' in c or 'time' in c or 'period' in c]
    value_cols = [c for c in df.columns if 'value' in c or 'actual' in c or 'index' in c]
    
    if not date_cols or not value_cols:
        return None
    
    df['date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    df['value'] = pd.to_numeric(df[value_cols[0]], errors='coerce')
    
    df = df.dropna(subset=['date', 'value'])
    df = df[['date', 'value']].rename(columns={'value': name or 'value'})
    return df

def compute_lag_correlation(series_a, series_b, max_lag=12, min_periods=12):
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            shifted_a = series_a.shift(-lag)
            shifted_b = series_b
        else:
            shifted_a = series_a
            shifted_b = series_b.shift(lag)

        combined = pd.concat([shifted_a, shifted_b], axis=1).dropna()
        if len(combined) >= min_periods:
            corr = combined.corr().iloc[0, 1]
            correlations.append((lag, corr))
    return pd.DataFrame(correlations, columns=["Lag", "Correlation"])

def create_correlation_heatmap(merged_df):
    corr_matrix = merged_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        colorbar=dict(title='Correlation')
    )) 
    fig.update_layout(
        title='Indicator Correlation Matrix',
        xaxis_title="Indicators",
        yaxis_title="Indicators",
        height=600
    )
    return fig

def plot_seasonality(df, col_name):
    df = df.copy()
    df['month'] = df['date'].dt.month_name()
    df['month'] = pd.Categorical(df['month'], categories=list(calendar.month_name)[1:], ordered=True)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Monthly Distribution', 'Annual Trend'))
    
    # Box plot
    fig.add_trace(go.Box(
        x=df['month'],
        y=df[col_name],
        name='Distribution',
        boxmean=True,
        marker_color='#3498db'
    ), row=1, col=1)
    
    # Annual trend
    df['year'] = df['date'].dt.year
    yearly = df.groupby('year')[col_name].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly[col_name],
        mode='lines+markers',
        name='Annual Trend',
        line=dict(color='#e74c3c')
    ), row=1, col=2)
    
    fig.update_layout(
        title=f'Seasonality Analysis: {col_name}',
        showlegend=False,
        height=400
    )
    return fig

def plot_lead_lag(lag_df, indicator1, indicator2):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lag_df['Lag'],
        y=lag_df['Correlation'],
        mode='lines+markers',
        name='Correlation',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8)
    )
    
    max_corr_row = lag_df.iloc[lag_df['Correlation'].abs().argmax()]
    fig.add_vline(
        x=max_corr_row['Lag'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Peak correlation (lag {max_corr_row['Lag']})",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f'Lead-Lag Relationship: {indicator1} vs {indicator2}',
        xaxis_title='Lag (months)',
        yaxis_title='Correlation Coefficient',
        hovermode="x unified",
        height=500,
        template="plotly_white"
    )
    return fig

# === Main App Logic ===
if uploaded_files:
    # Process uploaded files
    data_dict = {}
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            name = file.name.replace(".csv", "")
            processed_df = preprocess_df(df, name)
            if processed_df is not None:
                data_dict[name] = processed_df
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    
    if not data_dict:
        st.warning("No valid data found in uploaded files. Please check your file formats.")
    else:
        # Merge all data
        merged_df = None
        for name, df in data_dict.items():
            df = df.rename(columns={'value': name})
            merged_df = df if merged_df is None else pd.merge(merged_df, df, on='date', how='outer')
        merged_df = merged_df.set_index('date').sort_index()
        
        # Display data summary
        with st.expander("📁 Dataset Overview", expanded=True):
            cols = st.columns(3)
            cols[0].metric("Total Indicators", len(data_dict))
            cols[1].metric("Time Period", 
                          f"{merged_df.index.min().strftime('%Y-%m-%d')} to {merged_df.index.max().strftime('%Y-%m-%d')}")
            cols[2].metric("Total Observations", len(merged_df))
            
            selected_indicator = st.selectbox("Select indicator to preview", list(data_dict.keys()))
            st.dataframe(data_dict[selected_indicator].head(), use_container_width=True)
        
        # Analysis sections
        if analysis_type == "📈 Overview":
            st.header("📊 Data Overview")
            
            # Time series plot
            indicators_to_plot = st.multiselect(
                "Select indicators to plot", 
                list(data_dict.keys()),
                default=list(data_dict.keys())[:2]
            )
            
            if indicators_to_plot:
                fig = go.Figure()
                for indicator in indicators_to_plot:
                    fig.add_trace(go.Scatter(
                        x=merged_df.index,
                        y=merged_df[indicator],
                        name=indicator,
                        mode='lines',
                        hovertemplate="%{x|%b %Y}<br>%{y:.2f}<extra></extra>"
                    ))
                
                fig.update_layout(
                    title="Indicator Time Series",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode="x unified",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            st.header("🔗 Indicator Correlations")
            st.plotly_chart(create_correlation_heatmap(merged_df), use_container_width=True)
            
        elif analysis_type == "🔗 Correlation":
            st.header("🔍 Detailed Correlation Analysis")
            
            # Lag correlation heatmap
            st.subheader("Lag Correlation Matrix")
            analyze_tier1_lag_correlation_streamlit(data_dict)
            
            # Scatter plot matrix
            st.subheader("Scatter Plot Matrix")
            selected_for_scatter = st.multiselect(
                "Select indicators for scatter matrix",
                list(data_dict.keys()),
                default=list(data_dict.keys())[:4]
            )
            
            if len(selected_for_scatter) >= 2:
                scatter_df = merged_df[selected_for_scatter].dropna()
                fig = px.scatter_matrix(
                    scatter_df,
                    dimensions=selected_for_scatter,
                    height=800,
                    title="Pairwise Relationships"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "🔄 Seasonality":
            st.header("📅 Seasonality Analysis")
            
            selected_indicator = st.selectbox(
                "Select indicator for seasonality analysis",
                list(data_dict.keys())
            )
            
            st.plotly_chart(
                plot_seasonality(data_dict[selected_indicator], selected_indicator),
                use_container_width=True
            )
            
            # Decomposition plot
            st.subheader("Time Series Decomposition")
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                decomposition = seasonal_decompose(
                    merged_df[selected_indicator].dropna(),
                    period=12,
                    model='additive'
                )
                
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=decomposition.observed.index,
                        y=decomposition.observed,
                        name='Observed'
                    ), row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=decomposition.trend.index,
                        y=decomposition.trend,
                        name='Trend'
                    ), row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=decomposition.seasonal.index,
                        y=decomposition.seasonal,
                        name='Seasonal'
                    ), row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=decomposition.resid.index,
                        y=decomposition.resid,
                        name='Residual'
                    ), row=4, col=1
                )
                
                fig.update_layout(
                    height=800,
                    showlegend=False,
                    title_text="Time Series Decomposition"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not perform decomposition: {str(e)}")
        
        elif analysis_type == "⏩ Lead-Lag":
            st.header("⏱️ Lead-Lag Analysis")
            
            col1, col2 = st.columns(2)
            indicator1 = col1.selectbox(
                "First Indicator",
                list(data_dict.keys()),
                index=0
            )
            indicator2 = col2.selectbox(
                "Second Indicator",
                list(data_dict.keys()),
                index=min(1, len(data_dict.keys())-1)
            )
            
            if indicator1 != indicator2:
                s1 = merged_df[indicator1]
                s2 = merged_df[indicator2]
                
                lag_df = compute_lag_correlation(s1, s2, max_lag=max_lag)
                
                if not lag_df.empty:
                    st.plotly_chart(
                        plot_lead_lag(lag_df, indicator1, indicator2),
                        use_container_width=True
                    )
                    
                    max_row = lag_df.iloc[lag_df['Correlation'].abs().argmax()]
                    cols = st.columns(3)
                    cols[0].metric(
                        "Maximum Correlation",
                        f"{max_row['Correlation']:.2f}",
                        f"at lag {max_row['Lag']}"
                    )
                    cols[1].metric(
                        "Positive Peak Lag",
                        f"{lag_df.iloc[lag_df['Correlation'].argmax()]['Lag']}",
                        f"Corr: {lag_df.iloc[lag_df['Correlation'].argmax()]['Correlation']:.2f}"
                    )
                    cols[2].metric(
                        "Negative Peak Lag",
                        f"{lag_df.iloc[lag_df['Correlation'].argmin()]['Lag']}",
                        f"Corr: {lag_df.iloc[lag_df['Correlation'].argmin()]['Correlation']:.2f}"
                    )
                else:
                    st.warning("Insufficient data for lead-lag analysis")
            else:
                st.warning("Please select two different indicators")
else:
    st.info("ℹ️ Please upload one or more CSV files to begin analysis")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             use_column_width=True)

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: #7f8c8d;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    <p>Economic Indicator Analyzer • Created with Streamlit • Data updates {datetime.now().strftime('%Y-%m-%d')}</p>
</div>
""", unsafe_allow_html=True)
