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

# Dark Mode CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Text colors */
    .st-b7, .st-c0, .st-c1, .st-c2, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9, .stMarkdown, .stText {
        color: #ffffff !important;
    }
    
    /* Cards and containers */
    .metric-card {
        background: #2d2d2d;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        color: #ffffff;
    }
    
    .tab-content {
        background: #2d2d2d;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: #ffffff;
    }
    
    /* Input widgets */
    .stTextInput, .stNumberInput, .stSelectbox, .stMultiselect, .stSlider, .stRadio {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border-color: #444444 !important;
    }
    
    /* Buttons */
    .st-eb {
        background-color: #4a90e2 !important;
        color: white !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1oe5cao {
        background-color: #1a1a1a !important;
    }
    
    /* Footer */
    .footer {
        color: #aaaaaa !important;
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
def plot_normalized_actuals(data_dict, selected_indicators):
    fig = go.Figure()
    
    for indicator in selected_indicators:
        df = data_dict[indicator].copy()
        
        try:
            # Min-max normalization to [0, 1]
            actual_min = df['value'].min()
            actual_max = df['value'].max()
            
            if actual_max == actual_min:
                st.warning(f"No variation in values for {indicator} - skipping normalization")
                continue

            normalized = (df['value'] - actual_min) / (actual_max - actual_min)

            fig.add_trace(go.Scatter(
                x=df['date'],
                y=normalized,
                mode='lines+markers',
                name=indicator,
                marker=dict(size=6),
                line=dict(width=1.5),
                opacity=0.8
            ))

        except Exception as e:
            st.error(f"Error processing {indicator}: {str(e)}")
            continue

    if len(fig.data) == 0:
        st.warning("No valid data available for normalized plot")
        return None

    fig.update_layout(
        title="Normalized Indicator Values (0 to 1 scale)",
        xaxis_title="Date",
        yaxis_title="Normalized Value",
        hovermode="x unified",
        height=500,
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='#444444',
            tickformat='%Y-%m'
        ),
        yaxis=dict(
            gridcolor='#444444',
            range=[0, 1]  # Fixed scale for comparison
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig
    
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
        height=600,
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#444444'),
        yaxis=dict(gridcolor='#444444')
    )
    return fig

def analyze_tier1_lag_correlation_streamlit(data_dict, min_periods=12, top_n=10, max_lag=12):
    """Analyze and visualize lag correlations between indicators"""
    processed = {}
    for name, df in data_dict.items():
        clean_df = preprocess_df(df)
        if clean_df is not None:
            processed[name] = clean_df

    if not processed:
        st.error("No valid data found.")
        return

    # Merge into one DataFrame
    merged_df = None
    for name, df in processed.items():
        df = df.rename(columns={'value': name})
        merged_df = df if merged_df is None else pd.merge(merged_df, df, on='date', how='outer')
    merged_df = merged_df.set_index('date')

    indicators = list(processed.keys())
    heatmap_matrix = pd.DataFrame(index=indicators, columns=indicators, dtype=float)
    annotations = pd.DataFrame(index=indicators, columns=indicators, dtype=str)
    lag_corr_dict = {}

    # Compute lag correlations
    for i in range(len(indicators)):
        for j in range(i + 1, len(indicators)):
            ind1 = indicators[i]
            ind2 = indicators[j]
            s1 = merged_df[ind1]
            s2 = merged_df[ind2]
            lag_df = compute_lag_correlation(s1, s2, max_lag=max_lag, min_periods=min_periods)
            if not lag_df.empty:
                max_row = lag_df.iloc[lag_df['Correlation'].abs().argmax()]
                corr = max_row['Correlation']
                lag = int(max_row['Lag'])
                lag_corr_dict[(ind1, ind2)] = (corr, lag)
                heatmap_matrix.loc[ind1, ind2] = corr
                heatmap_matrix.loc[ind2, ind1] = corr
                annotation = f"{corr:.2f}\nLag:{lag}"
                annotations.loc[ind1, ind2] = annotations.loc[ind2, ind1] = annotation

    # Heatmap
    st.subheader("📊 Lag Correlation Heatmap")
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_matrix.astype(float),
        x=heatmap_matrix.columns,
        y=heatmap_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=annotations,
        hoverongaps=False
    ))
    fig.update_layout(
        title="Maximum Correlation Coefficients and Lags",
        xaxis_title="Indicator",
        yaxis_title="Indicator",
        height=600,
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top Pairs Table - Simplified version
    st.subheader(f"🔝 Top {top_n} Correlated Pairs")
    top_pairs = sorted(lag_corr_dict.items(), key=lambda x: abs(x[1][0]), reverse=True)[:top_n]
    
    # Create a simple DataFrame without styling
    top_pairs_df = pd.DataFrame(
        [(a, b, f"{corr:.2f}", lag) for (a, b), (corr, lag) in top_pairs],
        columns=["Indicator 1", "Indicator 2", "Correlation", "Lag"]
    )
    
    # Display the table with Streamlit's native table display
    st.dataframe(top_pairs_df, use_container_width=True)

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
        marker_color='#4a90e2'
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
        height=400,
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white')
    )
    return fig

def plot_lead_lag(lag_df, indicator1, indicator2):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lag_df['Lag'],
        y=lag_df['Correlation'],
        mode='lines+markers',
        name='Correlation',
        line=dict(color='#4a90e2', width=2),
        marker=dict(size=8)
    ))
    
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
        template="plotly_dark",
        paper_bgcolor='#2d2d2d',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white')
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
                # Original values plot
                st.subheader("Original Values")
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
                    ),
                    paper_bgcolor='#2d2d2d',
                    plot_bgcolor='#2d2d2d',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Normalized plot
                st.subheader("Normalized Comparison")
                normalized_fig = plot_normalized_actuals(data_dict, indicators_to_plot)
                if normalized_fig:
                    st.plotly_chart(normalized_fig, use_container_width=True)
            
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
                fig.update_layout(
                    paper_bgcolor='#2d2d2d',
                    plot_bgcolor='#2d2d2d',
                    font=dict(color='white')
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
                    title_text="Time Series Decomposition",
                    paper_bgcolor='#2d2d2d',
                    plot_bgcolor='#2d2d2d',
                    font=dict(color='white')
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
             use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: #aaaaaa !important;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    <p>Economic Indicator Analyzer • Created with Streamlit • Data updates {datetime.now().strftime('%Y-%m-%d')}</p>
</div>
""", unsafe_allow_html=True)
