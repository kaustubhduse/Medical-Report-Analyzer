import pandas as pd
import plotly.express as px
import streamlit as st

def plot_metric_trend(historical_df, metric, date_col='date'):
    """Plots the time series trend for a single metric."""
    if metric not in historical_df.columns or date_col not in historical_df.columns:
        st.warning(f"Data for '{metric}' or date column '{date_col}' not found.")
        return

    # Ensure the date column is sorted
    df = historical_df[[date_col, metric]].dropna().sort_values(by=date_col)
    
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

    df = historical_df[[date_col, metric]].dropna()
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
    
    # --- THIS IS THE FIX ---
    # Ensure the date column exists before proceeding
    if date_col not in historical_df.columns:
        st.error(f"Date column '{date_col}' not found in the historical data. Cannot plot trends.")
        return

    for metric in metrics:
        plot_metric_trend(historical_df, metric, date_col)
        anomalies = detect_anomalies(historical_df, metric, date_col)
        if anomalies:
            # Format anomalies for better readability
            formatted_anomalies = [f"{date}: {value:.2f}" for date, value in anomalies]
            st.warning(f"Potential anomalies detected in **{metric}**: {'; '.join(formatted_anomalies)}")
