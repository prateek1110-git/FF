import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import numpy as np
import math
from io import StringIO
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 Economic Indicator Analysis Dashboard")

# Upload multiple files
uploaded_files = st.file_uploader("Upload one or more CSV files", type=['csv'], accept_multiple_files=True)

# --- Core Functions ---

def preprocess_df(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if 'reference_period' not in df.columns or 'actual' not in df.columns:
        return None
    df['reference_period'] = pd.to_datetime(df['reference_period'], errors='coerce')
    df = df.dropna(subset=['reference_period', 'actual'])
    df = df[['reference_period', 'actual']].rename(columns={'actual': 'value'})
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

def analyze_tier1_lag_correlation_streamlit(data_dict, min_periods=12, top_n=10, max_lag=12):
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
        merged_df = df if merged_df is None else pd.merge(merged_df, df, on='reference_period', how='outer')
    merged_df = merged_df.set_index('reference_period')

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
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    sns.heatmap(heatmap_matrix.astype(float), annot=annotations, fmt="", cmap="coolwarm", center=0, linewidths=0.5, ax=ax1)
    st.pyplot(fig1)

    # Top Pairs Plot
    top_pairs = sorted(lag_corr_dict.items(), key=lambda x: abs(x[1][0]), reverse=True)[:top_n]
    top_pairs_df = pd.DataFrame([(a, b, corr, lag) for (a, b), (corr, lag) in top_pairs],
                                columns=['Indicator 1', 'Indicator 2', 'Max Correlation', 'Lag'])

    st.subheader("📈 Top Lag Correlation Pairs")
    n = len(top_pairs_df)
    cols = 2
    rows = math.ceil(n / cols)
    fig2, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten()

    for i, row in top_pairs_df.iterrows():
        ind1 = row['Indicator 1']
        ind2 = row['Indicator 2']
        s1 = merged_df[ind1]
        s2 = merged_df[ind2]
        lag_df = compute_lag_correlation(s1, s2, max_lag=max_lag, min_periods=min_periods)

        ax = axes[i]
        ax.plot(lag_df['Lag'], lag_df['Correlation'], marker='o')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_title(f"{ind1} vs {ind2}\nMax Corr: {row['Max Correlation']:.2f} at Lag: {row['Lag']}", fontsize=10)
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Correlation")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig2.delaxes(axes[j])

    st.pyplot(fig2)

# === Main App Logic ===

if uploaded_files:
    data_dict = {}
    for file in uploaded_files:
        df = pd.read_csv(file)
        name = file.name.replace(".csv", "")
        data_dict[name] = df

    file_names = list(data_dict.keys())
    selected_file = st.selectbox("Select a file to preview", file_names)
    st.subheader(f"📄 Preview of {selected_file}")
    st.dataframe(data_dict[selected_file].head())

    analysis_type = st.selectbox("Select Analysis Type", ["Seasonality", "Correlation", "Lead-Lag"])

    df = data_dict[selected_file]

    if 'Release Date' in df.columns:
        df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        df = df.dropna(subset=['Release Date'])
        df['Month'] = df['Release Date'].dt.month_name()
        df['Month'] = pd.Categorical(df['Month'], categories=list(calendar.month_name)[1:], ordered=True)

        if analysis_type == "Seasonality":
            if 'Actual' in df.columns:
                st.subheader("📈 Seasonality Analysis (Actual Values)")
                fig, ax = plt.subplots(figsize=(14, 6))
                sns.boxplot(x='Month', y='Actual', data=df, ax=ax, showmeans=True,
                            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
                sns.swarmplot(x='Month', y='Actual', data=df, ax=ax, color='black', alpha=0.6)
                ax.axhline(0, linestyle='--', color='red')
                ax.set_title("Monthly Seasonality in Actual Values")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

        elif analysis_type == "Correlation":
            st.subheader("🔗 Lag-Based Correlation Analysis Across Files")
            analyze_tier1_lag_correlation_streamlit(data_dict)

        elif analysis_type == "Lead-Lag":
            st.subheader("⏩ Lead-Lag Analysis")
            cols = st.multiselect("Choose two columns to compare", options=df.columns)
            if len(cols) == 2:
                col1, col2 = cols
                lags = list(range(-12, 13))
                correlations = []
                for lag in lags:
                    shifted = df[col2].shift(lag)
                    corr = df[col1].corr(shifted)
                    correlations.append(corr)

                lag_df = pd.DataFrame({"Lag": lags, "Correlation": correlations})
                fig = px.line(lag_df, x='Lag', y='Correlation', title=f"Lead-Lag Correlation: {col1} vs {col2}")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'Release Date' column not found in the selected file.")
