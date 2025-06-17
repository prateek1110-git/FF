import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from scipy import stats
import plotly.express as px

st.set_page_config(layout="wide")

st.title("📊 Economic Indicator Analysis Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Date parsing
    if 'Release Date' in df.columns:
        df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        df = df.dropna(subset=['Release Date'])
        df['Month'] = df['Release Date'].dt.month_name()
        df['Month'] = pd.Categorical(df['Month'], categories=list(calendar.month_name)[1:], ordered=True)

        analysis_type = st.selectbox("Select Analysis Type", ["Seasonality", "Correlation", "Lead-Lag"])

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
            st.subheader("🔗 Correlation Matrix")
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

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
