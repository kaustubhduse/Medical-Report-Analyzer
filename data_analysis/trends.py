import pandas as pd
import plotly.express as px
import streamlit as st

def plot_metric_trend(historical_df, metric, date_col='date'):
    """Plots the time series trend for a single metric."""
    if metric not in historical_df.columns or date_col not in historical_df.columns:
        st.warning(f"Data for '{metric}' or date column '{date_col}' not found.")
        return

    # Ensure the date column is parsed as datetime
    historical_df[date_col] = pd.to_datetime(historical_df[date_col], errors="coerce")

    # Ensure values are numeric
    historical_df[metric] = pd.to_numeric(historical_df[metric], errors="coerce")

    # Drop rows with missing values and sort
    df = historical_df[[date_col, metric]].dropna().sort_values(by=date_col)
    
    if df.empty:
        st.info(f"No valid data to plot for '{metric}'.")
        return

    fig = px.line(
        df, 
        x=date_col, 
        y=metric, 
        markers=True,
        title=f"{metric} Trend Over Time",
        labels={metric: metric, date_col: "Date"}
    )
    st.plotly_chart(fig, use_container_width=True)

def detect_anomalies(historical_df, metric, date_col='date', threshold=2.0):
    """Simple anomaly detection using z-score."""
    if metric not in historical_df.columns or date_col not in historical_df.columns:
        return []

    # Ensure numeric conversion
    df = historical_df[[date_col, metric]].dropna().copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna()  # remove rows where value couldn't be converted

    if df.empty:
        return []

    values = df[metric]
    
    # Check if there's enough data to calculate standard deviation
    if len(values) < 2 or values.std() == 0:
        return []
        
    mean, std = values.mean(), values.std()
    anomalies = df[abs(values - mean) > threshold * std]
    return list(anomalies.itertuples(index=False, name=None))

def show_trend_analysis(historical_df, metrics, date_col='date'):
    """
    Streamlit dashboard for trends and anomaly alerts.
    Expects a 'wide' DataFrame where columns are metrics and rows are indexed by date.
    """
    st.subheader("Metric Trends Over Time")
    
    # Ensure the date column exists before proceeding
    if date_col not in historical_df.columns:
        st.error(f"Date column '{date_col}' not found in the historical data. Cannot plot trends.")
        return

    for metric in metrics:
        plot_metric_trend(historical_df, metric, date_col)
        anomalies = detect_anomalies(historical_df, metric, date_col)
        if anomalies:
            # Format anomalies for better readability
            formatted_anomalies = [f"{date.strftime('%Y-%m-%d')}: {value:.2f}" 
                                   for date, value in anomalies if pd.notna(date)]
            if formatted_anomalies:
                st.warning(f"⚠️ Potential anomalies detected in **{metric}**: {'; '.join(formatted_anomalies)}")
