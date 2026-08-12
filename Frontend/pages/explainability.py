"""Explainability & Attention Visualization Page."""
from __future__ import annotations
import cv2
import numpy as np
import streamlit as st
from utils.helper import display_image

def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h1 style="margin:0; font-size: 1.8rem; color: #F8FAFC;">Explainability & Feature Attention</h1>
            <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">Visual review of model saliency, Grad-CAM++ feature activations, and attention distributions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    result = st.session_state.get("result")
    if not result:
        st.info("No active prediction found. Run a prediction in the **Prediction** tab to inspect Grad-CAM++ feature attention maps.")
        return

    original = st.session_state.original
    heatmap_rgb = result["heatmap"]
    
    with st.container(border=True):
        st.subheader("Interactive Attention Overlay Scrubbing", divider="blue")
        alpha = st.slider("Heatmap Opacity Scrubbing (0% = Original Image, 100% = Heatmap)", 0.0, 1.0, 0.45, 0.05)
        
        # Interactive blend
        blended = cv2.addWeighted(original, 1.0 - alpha, heatmap_rgb, alpha, 0)
        
        c1, c2 = st.columns(2)
        with c1:
            display_image(f"Blended Saliency Map ({int(alpha*100)}% Heatmap)", blended)
        with c2:
            display_image("Raw Grad-CAM++ Heatmap", heatmap_rgb)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Clinical Interpretability & Saliency Analysis", divider="blue")
        st.markdown(
            """
            - **High Intensity Regions (Red / Yellow):** Indicate ultrasound acoustic boundaries and parietal bone echoes contributing most heavily to the predicted segmentation mask.
            - **Low Intensity Regions (Blue / Cyan):** Indicate background tissue, amniotic fluid, or acoustic shadows suppressed by the model.
            - **Residual Attention Gating:** In architectures like `nnUNet_ResAttn`, attention gates filter out spurious high-frequency noise from soft tissue boundaries outside the skull.
            """
        )
        st.warning("For research and clinical decision support only. All automated outputs should be reviewed by a qualified sonographer.")
