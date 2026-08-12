"""Prediction & Fetal Biometry Workspace Page with Segmentation Error Analysis."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from model.preprocessing import prepare_image
from model.predictor import predict
from model.gradcam import generate_gradcam
from model.hc_measurement import estimate_head_circumference
from model.metrics import calculate_metrics
from utils.helper import card, display_image, metric_value
from utils.error_analysis import (
    compute_error_metrics,
    render_error_map_rgb,
    render_error_overlay,
    export_pixel_stats_csv,
    export_png_bytes,
)

PRESET_SAMPLES = {
    "Select a preset sample...": None,
    "Sample #301 (HC18 Validation)": "301_HC.png",
    "Sample #110 (HC18 Validation)": "110_HC.png",
    "Sample #103 (HC18 Validation)": "103_HC.png",
    "Sample #680 (HC18 Validation)": "680_3HC.png",
    "Sample #694 (HC18 Validation)": "694_HC.png",
    "Sample #125 (HC18 Validation)": "125_2HC.png",
}

VAL_IMG_DIR = Path(__file__).resolve().parent.parent.parent / "HC18" / "val" / "images"
VAL_MASK_DIR = Path(__file__).resolve().parent.parent.parent / "HC18" / "val" / "masks"
TEST_IMG_DIR = Path("C:/Users/shyle/OneDrive/Documents/Test/images")
TEST_MASK_DIR = Path("C:/Users/shyle/OneDrive/Documents/Test/mask")


def render_error_legend_html(stats: dict[str, float | int]) -> None:
    """Render publication-grade HTML color legend with pixel counts and percentages."""
    tp_cnt, tp_pct = stats["TP Pixels"], stats["TP Percent"]
    fp_cnt, fp_pct = stats["FP Pixels"], stats["FP Percent"]
    fn_cnt, fn_pct = stats["FN Pixels"], stats["FN Percent"]
    tn_cnt, tn_pct = stats["TN Pixels"], stats["TN Percent"]

    st.markdown(
        f"""
        <div class="error-legend-box">
            <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.6rem;">
                Pixel Confusion Legend & Classification Counts
            </div>
            <div class="legend-grid">
                <div class="legend-item" style="border-left: 3px solid #FFFFFF;">
                    <div class="legend-swatch" style="background: #FFFFFF; border: 1px solid #CBD5E1;"></div>
                    <div class="legend-info">
                        <span class="legend-title">True Positive (TP)</span>
                        <span class="legend-detail">GT = 1, Pred = 1</span>
                        <span class="legend-detail" style="color: #F8FAFC; font-weight: 600;">{tp_cnt:,} px ({tp_pct:.2f}%)</span>
                    </div>
                </div>
                <div class="legend-item" style="border-left: 3px solid #FF0000;">
                    <div class="legend-swatch" style="background: #FF0000;"></div>
                    <div class="legend-info">
                        <span class="legend-title">False Positive (FP)</span>
                        <span class="legend-detail">GT = 0, Pred = 1</span>
                        <span class="legend-detail" style="color: #F87171; font-weight: 600;">{fp_cnt:,} px ({fp_pct:.2f}%)</span>
                    </div>
                </div>
                <div class="legend-item" style="border-left: 3px solid #00FF00;">
                    <div class="legend-swatch" style="background: #00FF00;"></div>
                    <div class="legend-info">
                        <span class="legend-title">False Negative (FN)</span>
                        <span class="legend-detail">GT = 1, Pred = 0</span>
                        <span class="legend-detail" style="color: #34D399; font-weight: 600;">{fn_cnt:,} px ({fn_pct:.2f}%)</span>
                    </div>
                </div>
                <div class="legend-item" style="border-left: 3px solid #64748B;">
                    <div class="legend-swatch" style="background: #000000; border: 1px solid #475569;"></div>
                    <div class="legend-info">
                        <span class="legend-title">True Negative (TN)</span>
                        <span class="legend-detail">GT = 0, Pred = 0</span>
                        <span class="legend-detail" style="color: #94A3B8; font-weight: 600;">{tn_cnt:,} px ({tn_pct:.2f}%)</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h1 style="margin:0; font-size: 1.8rem; color: #F8FAFC;">Prediction & Fetal Biometry Workspace</h1>
            <p style="color: #94A3B8; margin: 0.25rem 0 0 0;">Upload an ultrasound image or choose a preset test frame to run AI segmentation, biometry estimation, and pixel-wise error analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input Selection Container
    with st.container(border=True):
        st.subheader("Select Ultrasound Test Image", divider="blue")

        mode_options = ["Upload Ultrasound Image", "Select HC18 Validation Preset"]
        if TEST_IMG_DIR.exists():
            mode_options.append("Select Sample from Local Test Directory")

        mode = st.radio("Input Source", mode_options, horizontal=True)

        source_image = None
        ground_truth_mask = None

        if mode == "Upload Ultrasound Image":
            col1, col2 = st.columns(2)
            with col1:
                uploaded_img = st.file_uploader("Upload Ultrasound Image (.png, .jpg, .bmp)", type=["png", "jpg", "jpeg", "bmp"], key="upload_img")
                if uploaded_img:
                    source_image = Image.open(uploaded_img)
            with col2:
                uploaded_mask = st.file_uploader("Ground-Truth Mask (Optional)", type=["png", "jpg", "jpeg", "bmp"], key="upload_gt")
                if uploaded_mask:
                    ground_truth_mask = np.array(Image.open(uploaded_mask).convert("L"))
        elif mode == "Select HC18 Validation Preset":
            preset_choice = st.selectbox("Select Preset Validation Sample", list(PRESET_SAMPLES.keys()))
            if preset_choice and PRESET_SAMPLES[preset_choice]:
                fn = PRESET_SAMPLES[preset_choice]
                img_path = VAL_IMG_DIR / fn
                if img_path.exists():
                    source_image = Image.open(img_path)

                stem = fn.rsplit(".", 1)[0]
                mask_path = VAL_MASK_DIR / f"{stem}_Annotation.png"
                if mask_path.exists():
                    ground_truth_mask = np.array(Image.open(mask_path).convert("L"))
        else:
            test_files = [f.name for f in TEST_IMG_DIR.glob("*.png")]
            test_choice = st.selectbox("Select Test Image", ["Select a test image..."] + test_files)
            if test_choice != "Select a test image...":
                img_path = TEST_IMG_DIR / test_choice
                if img_path.exists():
                    source_image = Image.open(img_path)

                stem = test_choice.rsplit(".", 1)[0]
                mask_candidates = [
                    TEST_MASK_DIR / f"{stem}_Annotation.png",
                    TEST_MASK_DIR / test_choice,
                ]
                for mc in mask_candidates:
                    if mc.exists():
                        ground_truth_mask = np.array(Image.open(mc).convert("L"))
                        break

        # Process selected image for display preview
        if source_image:
            _, rgb = prepare_image(source_image, st.session_state.image_size)
            if "original" not in st.session_state or not np.array_equal(st.session_state.original, rgb):
                st.session_state.original = rgb
                st.session_state.pop("result", None)

            if ground_truth_mask is not None:
                contours, _ = cv2.findContours(ground_truth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours and np.count_nonzero(ground_truth_mask) < (ground_truth_mask.size * 0.10):
                    filled_gt = np.zeros_like(ground_truth_mask)
                    cv2.drawContours(filled_gt, contours, -1, 255, -1)
                    ground_truth_mask = filled_gt
                gt_resized = cv2.resize(ground_truth_mask, (st.session_state.image_size, st.session_state.image_size), interpolation=cv2.INTER_NEAREST)
                st.session_state.truth = gt_resized
            else:
                st.session_state.pop("truth", None)

            st.markdown("<br>", unsafe_allow_html=True)
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                display_image("Input Ultrasound Image Preview", rgb, f"Resolution: {rgb.shape[1]}x{rgb.shape[0]} px")
            with p_col2:
                if "truth" in st.session_state:
                    display_image("Ground-Truth Reference Mask", st.session_state.truth, "Loaded reference annotation")
                else:
                    st.info("No ground-truth mask uploaded. AI will run prediction without reference comparison.")

    # Model status guidance
    details = st.session_state.get("model_details")
    if details and details.is_uploaded:
        st.success(f"🟢 **Active Model:** {details.name} | Size: {details.size}")
    else:
        st.warning("⚠️ **No model checkpoint uploaded.** Please upload a model file (`.pth` / `.pt`) under **Model Settings** in the sidebar to enable PyTorch inference.")

    # Action Button
    if "original" in st.session_state:
        if st.button("Run AI Segmentation & Test Image", type="primary", width="stretch", icon=":material/play_circle:"):
            with st.spinner("Processing Ultrasound Image & Computing Fetal Biometry..."):
                try:
                    mask, confidence, elapsed = predict(
                        st.session_state.model_details.model,
                        st.session_state.original,
                        st.session_state.threshold,
                    )
                    overlay, heatmap = generate_gradcam(
                        st.session_state.model_details.model,
                        st.session_state.original,
                        mask,
                    )
                    hc_res = estimate_head_circumference(
                        st.session_state.original,
                        mask,
                        pixel_spacing_mm=st.session_state.pixel_spacing,
                    )
                    truth = st.session_state.get("truth")
                    metrics = calculate_metrics(mask, truth)

                    error_metrics = None
                    if truth is not None:
                        error_metrics = compute_error_metrics(mask, truth)

                    st.session_state.result = {
                        "mask": mask,
                        "confidence": confidence,
                        "elapsed": elapsed,
                        "gradcam": overlay,
                        "heatmap": heatmap,
                        "hc": hc_res,
                        "metrics": metrics,
                        "error_metrics": error_metrics,
                    }
                    st.success(f"Segmentation & Biometry Test Completed in {elapsed*1000:.1f} ms!")
                except Exception as exc:
                    st.error(f"Prediction error: {exc}")

    # Display Results ONLY AFTER User Runs Prediction
    if result := st.session_state.get("result"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Clinical Test Results & Biometry")

        hc = result["hc"]

        # Clinical Biometry KPI Cards
        cols = st.columns(4)
        with cols[0]:
            card("Head Circumference (HC)", f"{hc.circumference_mm:.1f} mm" if hc.circumference_mm else "--", "Fitted ellipse perimeter", style="accent")
        with cols[1]:
            card("Gestational Age (GA)", hc.gestational_age_str, "Hadlock Formula estimate", style="success")
        with cols[2]:
            card("Biparietal Diam (BPD)", f"{hc.bpd_mm:.1f} mm" if hc.bpd_mm else "--", "Minor axis length", style="default")
        with cols[3]:
            card("Occipitofrontal Diam (OFD)", f"{hc.ofd_mm:.1f} mm" if hc.ofd_mm else "--", "Major axis length", style="default")

        st.markdown("<br>", unsafe_allow_html=True)

        # Multi-view Image Visualizer Tabs
        st.markdown("#### Segmentation & Analysis Panels")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Quad-View Comparison",
            "Segmentation Error Analysis",
            "Contour & Ellipse Fitting",
            "Attention Heatmap (Grad-CAM++)",
        ])

        truth_img = st.session_state.get("truth")
        pred_mask = result["mask"]
        orig_img = st.session_state.original

        # ----------------------------------------------------
        # TAB 1: QUAD-VIEW COMPARISON
        # ----------------------------------------------------
        with tab1:
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                display_image("Original Ultrasound Image", orig_img, f"{orig_img.shape[1]}x{orig_img.shape[0]} px")
            with r1c2:
                gt_display = truth_img if truth_img is not None else np.zeros_like(pred_mask)
                display_image("Ground-Truth Mask", gt_display, "Reference annotation" if truth_img is not None else "No GT provided")

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                display_image("Predicted Head Mask", pred_mask, f"Confidence: {result['confidence']*100:.1f}%")
            with r2c2:
                if truth_img is not None:
                    pure_err = render_error_map_rgb(pred_mask, truth_img)
                    display_image("Segmentation Error Map (TP/TN/FP/FN)", pure_err, "⚪ TP | 🔴 FP | 🟢 FN | ⚫ TN")
                else:
                    display_image("Fitted Ellipse & HC Overlay", hc.visualization, f"HC: {hc.circumference_mm:.1f} mm" if hc.circumference_mm else "Contour fitted")

        # ----------------------------------------------------
        # TAB 2: SEGMENTATION ERROR ANALYSIS (DEDICATED PANEL)
        # ----------------------------------------------------
        with tab2:
            st.subheader("Segmentation Error Analysis (TP / TN / FP / FN)", divider="blue")

            if truth_img is None:
                st.warning("⚠️ **Ground-Truth Mask Required**: Upload a ground-truth mask or select an HC18 validation sample to perform pixel-wise error analysis.")
            else:
                error_stats = result.get("error_metrics") or compute_error_metrics(pred_mask, truth_img)

                # Interactive Controls Bar inside a container
                with st.container(border=True):
                    st.markdown("##### Interactive Controls & Visualization Modes")
                    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.2, 1.2, 1.6])

                    with ctrl_c1:
                        overlay_mode = st.radio(
                            "Overlay Base Image",
                            ["Original Ultrasound Image", "Predicted Mask", "Pure Error Map"],
                            index=0,
                            key="err_overlay_mode",
                        )

                    with ctrl_c2:
                        alpha_val = st.slider(
                            "Overlay Transparency",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.55,
                            step=0.05,
                            key="err_alpha_slider",
                            disabled=(overlay_mode == "Pure Error Map"),
                        )

                    with ctrl_c3:
                        st.write("Toggle Class Visibility:")
                        tc1, tc2 = st.columns(2)
                        with tc1:
                            vis_tp = st.checkbox("⚪ True Positive (TP)", value=True, key="vis_tp")
                            vis_fp = st.checkbox("🔴 False Positive (FP)", value=True, key="vis_fp")
                        with tc2:
                            vis_fn = st.checkbox("🟢 False Negative (FN)", value=True, key="vis_fn")
                            vis_tn = st.checkbox("⚫ True Negative (TN)", value=False, key="vis_tn")

                # Generate rendered map according to user controls
                pure_error_rgb = render_error_map_rgb(
                    pred_mask,
                    truth_img,
                    vis_tp=vis_tp,
                    vis_fp=vis_fp,
                    vis_fn=vis_fn,
                    vis_tn=vis_tn,
                )

                if overlay_mode == "Pure Error Map":
                    rendered_image = pure_error_rgb
                    caption_str = "Pure Pixel-Wise Confusion Map"
                elif overlay_mode == "Original Ultrasound Image":
                    rendered_image = render_error_overlay(
                        orig_img,
                        pred_mask,
                        truth_img,
                        alpha=alpha_val,
                        vis_tp=vis_tp,
                        vis_fp=vis_fp,
                        vis_fn=vis_fn,
                        vis_tn=vis_tn,
                    )
                    caption_str = f"Error Map Overlaid on Original Ultrasound (Opacity: {int(alpha_val*100)}%)"
                else:  # Predicted Mask
                    pred_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
                    rendered_image = render_error_overlay(
                        pred_rgb,
                        pred_mask,
                        truth_img,
                        alpha=alpha_val,
                        vis_tp=vis_tp,
                        vis_fp=vis_fp,
                        vis_fn=vis_fn,
                        vis_tn=vis_tn,
                    )
                    caption_str = f"Error Map Overlaid on Predicted Mask (Opacity: {int(alpha_val*100)}%)"

                st.markdown("<br>", unsafe_allow_html=True)

                # Display Error Map inside a Rounded Card Container
                with st.container(border=True):
                    st.subheader("Pixel-Wise Confusion Map Visualization", divider="blue")
                    st.image(rendered_image, caption=caption_str, width="stretch")

                    # Render Color Legend below visualization
                    render_error_legend_html(error_stats)

                st.markdown("<br>", unsafe_allow_html=True)

                # Quantitative Pixel Statistics & Metrics Display
                with st.container(border=True):
                    st.subheader("Quantitative Pixel Statistics & Performance Metrics", divider="blue")

                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    with m_col1:
                        card("TP Pixels", f"{error_stats['TP Pixels']:,}", f"{error_stats['TP Percent']:.2f}% of total image", style="default")
                    with m_col2:
                        card("FP Pixels", f"{error_stats['FP Pixels']:,}", f"{error_stats['FP Percent']:.2f}% of total image", style="warning")
                    with m_col3:
                        card("FN Pixels", f"{error_stats['FN Pixels']:,}", f"{error_stats['FN Percent']:.2f}% of total image", style="success")
                    with m_col4:
                        card("TN Pixels", f"{error_stats['TN Pixels']:,}", f"{error_stats['TN Percent']:.2f}% of total image", style="default")

                    st.markdown("<br>", unsafe_allow_html=True)

                    met_c1, met_c2, met_c3, met_c4 = st.columns(4)
                    with met_c1:
                        card("Pixel Accuracy", f"{error_stats['Pixel Accuracy']:.4f}", "(TP + TN) / Total", style="accent")
                        card("False Positive Rate", f"{error_stats['False Positive Rate']:.4f}", "FP / (FP + TN)", style="default")
                    with met_c2:
                        card("Sensitivity (Recall)", f"{error_stats['Sensitivity (Recall)']:.4f}", "TP / (TP + FN)", style="accent")
                        card("False Negative Rate", f"{error_stats['False Negative Rate']:.4f}", "FN / (TP + FN)", style="default")
                    with met_c3:
                        card("Specificity", f"{error_stats['Specificity']:.4f}", "TN / (TN + FP)", style="accent")
                        card("Dice Score (F1)", f"{error_stats['Dice Score']:.4f}", "2*TP / (2*TP + FP + FN)", style="success")
                    with met_c4:
                        card("Precision", f"{error_stats['Precision']:.4f}", "TP / (TP + FP)", style="accent")
                        card("IoU (Jaccard)", f"{error_stats['IoU']:.4f}", "TP / (TP + FP + FN)", style="success")

                st.markdown("<br>", unsafe_allow_html=True)

                # Export / Download Options Container
                with st.container(border=True):
                    st.subheader("Publication Figures & Data Export", divider="blue")
                    st.write("Download publication-quality figures and pixel statistics for research papers and presentations.")

                    d_col1, d_col2, d_col3 = st.columns(3)

                    # 1. Error Map (PNG)
                    error_map_png = export_png_bytes(pure_error_rgb)
                    with d_col1:
                        st.download_button(
                            label="Download Error Map (PNG)",
                            data=error_map_png,
                            file_name="segmentation_error_map.png",
                            mime="image/png",
                            width="stretch",
                            icon=":material/image:",
                        )

                    # 2. Overlay Image (PNG)
                    overlay_png = export_png_bytes(rendered_image)
                    with d_col2:
                        st.download_button(
                            label="Download Overlay Image (PNG)",
                            data=overlay_png,
                            file_name="segmentation_error_overlay.png",
                            mime="image/png",
                            width="stretch",
                            icon=":material/layers:",
                        )

                    # 3. Pixel Statistics (CSV)
                    model_lbl = st.session_state.model_details.name if "model_details" in st.session_state else "nnUNet_ResAttn"
                    stats_csv = export_pixel_stats_csv(error_stats, model_name=model_lbl)
                    with d_col3:
                        st.download_button(
                            label="Download Pixel Statistics (CSV)",
                            data=stats_csv,
                            file_name="pixel_confusion_statistics.csv",
                            mime="text/csv",
                            width="stretch",
                            icon=":material/table_chart:",
                        )

        # ----------------------------------------------------
        # TAB 3: CONTOUR & ELLIPSE FITTING
        # ----------------------------------------------------
        with tab3:
            st.subheader("Largest Contour Extraction & Ellipse Fitting", divider="blue")
            display_image("Fitted Ellipse & Biometry Annotations", hc.visualization)

        # ----------------------------------------------------
        # TAB 4: ATTENTION HEATMAP (GRAD-CAM++)
        # ----------------------------------------------------
        with tab4:
            st.subheader("Grad-CAM++ Feature Activation Overlay", divider="blue")
            c1, c2 = st.columns(2)
            with c1:
                display_image("Grad-CAM++ Overlay", result["gradcam"], "Blended attention overlay")
            with c2:
                display_image("Raw Heatmap Map", result["heatmap"], "Intensity map")
    elif "original" not in st.session_state:
        st.info("👆 Please upload an ultrasound image or select a preset sample to begin testing.")
