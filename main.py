import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import calendar
from datetime import datetime

# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="📊 Economic Indicator Analyzer",
    page_icon="📈"
)

# Custom CSS Styling
st.markdown("""
<style>
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
st.markdown("Explore relationships between economic indicators through correlation, seasonality, and lead-lag analysis.")

# --- Core Functions ---
def preprocess_df(df, name=""):
    """Clean and standardize the input dataframe"""
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
    return df[['date', 'value']].rename(columns={'value': name or 'value'})

def create_correlation_heatmap(merged_df):
    """Create an interactive correlation heatmap"""
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
    )
    fig.update_layout(
        title='Indicator Correlation Matrix',
        xaxis_title="Indicators",
        yaxis_title="Indicators",
        height=600
    )
    return fig

def plot_seasonality(df, col_name):
    """Create seasonality visualization"""
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

# --- Main App Logic ---
uploaded_files = st.file_uploader(
    "📤 Upload CSV Files", 
    type=['csv'], 
    accept_multiple_files=True,
    help="Upload one or more CSV files containing economic indicator data"
)

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
    
    if data_dict:
        # Merge all data
        merged_df = pd.concat(
            [df.set_index('date') for df in data_dict.values()], 
            axis=1
        ).sort_index().dropna()
        
        # Display data summary
        with st.expander("📁 Dataset Overview", expanded=True):
            cols = st.columns(3)
            cols[0].metric("Total Indicators", len(data_dict))
            cols[1].metric(
                "Time Period", 
                f"{merged_df.index.min().strftime('%Y-%m-%d')} to {merged_df.index.max().strftime('%Y-%m-%d')}"
            )
            cols[2].metric("Total Observations", len(merged_df))
            
            st.plotly_chart(create_correlation_heatmap(merged_df), use_container_width=True)
        
        # Time series visualization
        st.header("📈 Time Series Visualization")
        selected_indicators = st.multiselect(
            "Select indicators to plot",
            list(data_dict.keys()),
            default=list(data_dict.keys())[:2]
        )
        
        if selected_indicators:
            fig = go.Figure()
            for indicator in selected_indicators:
                fig.add_trace(go.Scatter(
                    x=merged_df.index,
                    y=merged_df[indicator],
                    name=indicator,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Indicator Time Series",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonality analysis
        st.header("🔄 Seasonality Analysis")
        selected_indicator = st.selectbox(
            "Select indicator for seasonality analysis",
            list(data_dict.keys())
        )
        st.plotly_chart(
            plot_seasonality(data_dict[selected_indicator], selected_indicator),
            use_container_width=True
        )
else:
    st.info("ℹ️ Please upload one or more CSV files to begin analysis")

# Footer
st.markdown("---")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
