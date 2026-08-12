"""Metric dashboard, benchmarking charts, and PDF/CSV clinical report export."""
from __future__ import annotations
import plotly.graph_objects as go
import streamlit as st
from utils.helper import card, metric_value, report_csv, make_pdf

def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h1 style="margin:0; font-size: 1.8rem; color: #F8FAFC;">Metrics & Clinical Reports</h1>
            <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">Quantitative segmentation metrics, operational profile, and report generation.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    result = st.session_state.get("result")
    if not result:
        st.info("No active test result found. Upload an ultrasound image and click **Run AI Segmentation & Test Image** in the Prediction workspace first.")
        return

    metrics = result["metrics"].copy()
    details = st.session_state.model_details
    model_label = details.checkpoint_path.name if (details.is_uploaded and details.checkpoint_path) else "No Model Uploaded"
    metrics.update({
        "Inference Time (s)": round(result["elapsed"], 4),
        "FPS": round(1.0 / max(result["elapsed"], 0.001), 1),
        "Active Model": model_label,
    })
    if result["hc"].circumference_mm:
        metrics["HC Estimated (mm)"] = round(result["hc"].circumference_mm, 1)

    st.markdown("#### Quantitative Performance Profile")
    
    # Render KPI Cards Grid
    metric_keys = list(metrics.keys())
    for start in range(0, len(metric_keys), 4):
        cols = st.columns(4)
        for col, key in zip(cols, metric_keys[start:start+4]):
            with col:
                card(key, metric_value(metrics[key]), style="accent" if "Dice" in key or "HC" in key else "default")

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly Visualizations
    col_a, col_b = st.columns(2)
    
    # Radar Polar Chart for Segmentation Metrics
    radar_labels = ["Dice Coefficient", "IoU (Jaccard)", "Precision", "Recall (Sensitivity)", "F1 Score"]
    radar_vals = []
    for lbl in radar_labels:
        v = metrics.get(lbl, 0.0)
        radar_vals.append(v if isinstance(v, float) else 0.0)

    with col_a:
        with st.container(border=True):
            st.subheader("Segmentation Quality Radar", divider="blue")
            fig = go.Figure(
                go.Scatterpolar(
                    r=radar_vals + [radar_vals[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill="toself",
                    fillcolor="rgba(0, 102, 255, 0.25)",
                    line=dict(color="#0066FF", width=2),
                )
            )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                height=320,
                margin=dict(l=40, r=40, t=30, b=30),
            )
            st.plotly_chart(fig, width="stretch")

    # Confidence Gauge Chart
    with col_b:
        with st.container(border=True):
            st.subheader("Prediction Confidence Indicator", divider="blue")
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["confidence"] * 100,
                    number={"suffix": "%", "font": {"color": "#F8FAFC", "size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#CBD5E1"},
                        "bar": {"color": "#0066FF"},
                        "steps": [
                            {"range": [0, 50], "color": "rgba(239, 68, 68, 0.2)"},
                            {"range": [50, 80], "color": "rgba(245, 158, 11, 0.2)"},
                            {"range": [80, 100], "color": "rgba(16, 185, 129, 0.2)"},
                        ],
                    },
                )
            )
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                height=320,
                margin=dict(l=30, r=30, t=40, b=20),
            )
            st.plotly_chart(fig_g, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # Export Section
    with st.container(border=True):
        st.subheader("Export Clinical Reports", divider="blue")
        st.write("Export quantitative segmentation metrics and biometry measurements for documentation.")
        
        c1, c2 = st.columns(2)
        csv_data = report_csv(metrics, st.session_state.model_details.name)
        pdf_data = make_pdf(metrics, st.session_state.model_details.name)
        
        with c1:
            st.download_button(
                "Download CSV Metrics Data",
                csv_data,
                "fetal_hc_metrics.csv",
                "text/csv",
                width="stretch",
                icon=":material/download:",
            )
        with c2:
            if pdf_data:
                st.download_button(
                    "Download Formatted PDF Report",
                    pdf_data,
                    "fetal_hc_clinical_report.pdf",
                    "application/pdf",
                    width="stretch",
                    icon=":material/picture_as_pdf:",
                )
