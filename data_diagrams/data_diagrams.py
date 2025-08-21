import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

def plot_metric_comparison(metrics_df, silent=False):
    """Interactive bar chart with reference ranges. Returns fig if silent=True."""
    if metrics_df.empty:
        return None

    fig = px.bar(
        metrics_df,
        x='metric',
        y='value',
        color='status',
        color_discrete_map={
            'Low': '#FF6B6B',
            'Normal': '#51CF66',
            'High': '#FF922B'
        },
        labels={'value': 'Value', 'metric': 'Metric'},
        title='Medical Metrics Analysis'
    )

    # Add reference range bands
    for idx, row in metrics_df.iterrows():
        if '-' in str(row['reference_range']):
            try:
                low, high = map(float, row['reference_range'].split('-'))
                fig.add_shape(
                    type="rect",
                    x0=idx-0.4, x1=idx+0.4,
                    y0=low, y1=high,
                    line=dict(color="RoyalBlue"),
                    fillcolor="LightSkyBlue",
                    opacity=0.3
                )
            except Exception:
                pass

    fig.update_layout(
        xaxis_tickangle=-45,
        hovermode="x unified",
        showlegend=False
    )
    if not silent:
        st.plotly_chart(fig, use_container_width=True)
    return fig

def generate_radial_health_score(metrics_df):
    """Radial chart for quick health assessment"""
    if metrics_df.empty:
        return

    normalized_df = metrics_df.copy()
    normalized_df['score'] = 0.0

    for idx, row in normalized_df.iterrows():
        if '-' in str(row['reference_range']):
            try:
                low, high = map(float, row['reference_range'].split('-'))
                if row['value'] < low:
                    normalized_df.at[idx, 'score'] = 0.2
                elif row['value'] > high:
                    normalized_df.at[idx, 'score'] = 0.8
                else:
                    normalized_df.at[idx, 'score'] = 0.5
            except Exception:
                normalized_df.at[idx, 'score'] = 0.5

    fig = px.line_polar(
        normalized_df,
        r='score',
        theta='metric',
        line_close=True,
        color_discrete_sequence=['#2ecc71'],
        title='Health Metric Balance'
    )
    st.plotly_chart(fig, use_container_width=True)

def create_clinical_summary_pdf(metrics_df):
    """Generate PDF report with visualizations"""
    from fpdf import FPDF
    import tempfile

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add table
    cols = ["Metric", "Value", "Reference", "Status"]
    pdf.cell(200, 10, txt="Clinical Report Summary", ln=1, align='C')

    # Create table
    col_widths = [45, 35, 60, 40]
    for col, width in zip(cols, col_widths):
        pdf.cell(width, 10, txt=col, border=1)
    pdf.ln()

    for _, row in metrics_df.iterrows():
        pdf.cell(col_widths[0], 10, txt=str(row['metric']), border=1)
        pdf.cell(col_widths[1], 10, txt=str(row['value']), border=1)
        pdf.cell(col_widths[2], 10, txt=str(row['reference_range']), border=1)
        pdf.cell(col_widths[3], 10, txt=str(row['status']), border=1)
        pdf.ln()

    # Add visualization
    fig = plot_metric_comparison(metrics_df, silent=True)
    if fig:
        img_bytes = fig.to_image(format="png")  # ✅ Works without Chrome
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            tmpfile.write(img_bytes)
            tmpfile.flush()
            pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=180)

    # Get PDF content as bytes
    pdf_content = pdf.output(dest='S').encode('latin1')
    return pdf_content



def display_reference_table(metrics_df):
    """Interactive reference table with filtering"""
    if metrics_df.empty:
        return

    st.subheader("📋 Interactive Reference Table")
    status_filter = st.multiselect(
        "Filter by Status",
        options=metrics_df['status'].unique(),
        default=metrics_df['status'].unique()
    )
    filtered_df = metrics_df[metrics_df['status'].isin(status_filter)]
    st.dataframe(
        filtered_df[['metric', 'value', 'unit', 'reference_range', 'status']],
        hide_index=True,
        use_container_width=True
    )

def plot_historical_trend(historical_data):
    """Line chart for historical metric trends"""
    if not historical_data:
        return

    fig = px.line(
        historical_data,
        x='date',
        y='value',
        color='metric',
        markers=True,
        title='Historical Metric Trends',
        labels={'value': 'Value', 'date': 'Date'}
    )

    for metric in historical_data['metric'].unique():
        ref_range = historical_data[historical_data['metric'] == metric].iloc[0]['reference_range']
        if '-' in str(ref_range):
            try:
                low, high = map(float, ref_range.split('-'))
                fig.add_hline(y=low, line_dash="dot", annotation_text=f"{metric} Lower Limit", line_color="red")
                fig.add_hline(y=high, line_dash="dot", annotation_text=f"{metric} Upper Limit", line_color="green")
            except Exception:
                pass

    st.plotly_chart(fig, use_container_width=True)
