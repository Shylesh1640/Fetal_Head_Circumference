"""About, system architecture, and model benchmarking documentation page."""
from __future__ import annotations
import streamlit as st
from model.model_loader import ARCHITECTURES

def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h1 style="margin:0; font-size: 1.8rem; color: #F8FAFC;">About Fetal Head Circumference AI</h1>
            <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">System architecture, dataset provenance, and deep learning benchmark comparisons.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.subheader("Processing Pipeline Architecture", divider="blue")
        
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; background: rgba(15,23,42,0.6); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); font-size: 0.85rem; font-weight: 600;">
                <div style="background: #0066FF; color: white; padding: 0.4rem 0.8rem; border-radius: 6px;">1. Ultrasound Input</div>
                <div style="color: #60A5FA;">➔</div>
                <div style="background: rgba(255,255,255,0.08); padding: 0.4rem 0.8rem; border-radius: 6px;">2. Preprocessing & Normalization</div>
                <div style="color: #60A5FA;">➔</div>
                <div style="background: #0066FF; color: white; padding: 0.4rem 0.8rem; border-radius: 6px;">3. Segmentation Model</div>
                <div style="color: #60A5FA;">➔</div>
                <div style="background: rgba(255,255,255,0.08); padding: 0.4rem 0.8rem; border-radius: 6px;">4. Ellipse Fitting</div>
                <div style="color: #60A5FA;">➔</div>
                <div style="background: #10B981; color: white; padding: 0.4rem 0.8rem; border-radius: 6px;">5. HC & GA Biometry</div>
                <div style="color: #60A5FA;">➔</div>
                <div style="background: rgba(255,255,255,0.08); padding: 0.4rem 0.8rem; border-radius: 6px;">6. Grad-CAM++ & Reports</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("7 Model Architecture Quantitative Comparison", divider="blue")
        st.write("Quantitative test evaluation across 7 deep learning segmentors trained and validated on HC18 ultrasound dataset.")
        
        table_data = []
        for key, val in ARCHITECTURES.items():
            table_data.append({
                "Architecture": val["name"],
                "Test Dice Score": f"{val['dice']:.4f}",
                "HD95 Distance": val["hd95"],
                "Parameters": val["params"],
                "Model Size": val["size"],
                "Key Architectural Feature": val["desc"],
            })
            
        st.dataframe(table_data, width="stretch", hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Dataset & Clinical Disclaimer", divider="blue")
        st.markdown(
            """
            - **Benchmark Dataset:** HC18 Grand Challenge (Fetal Head Circumference Challenge) comprising 999 2D ultrasound frames acquired across all trimesters of pregnancy.
            - **Formula Specifications:** Fetal Head Circumference (HC) calculated via Ramanujan's second approximation for ellipse perimeter. Gestational Age (GA) estimated via Hadlock's logarithmic regression formula.
            - **Notice:** This platform is intended solely for research, validation, and sonographic decision support.
            """
        )
        st.warning("All automated biometry measurements must be validated by a certified healthcare professional before clinical use.")
