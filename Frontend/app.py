"""Fetal Head Circumference AI System - Streamlit Entry Point."""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

# Explicitly insert Frontend path into sys.path to resolve local imports cleanly
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from streamlit_option_menu import option_menu
except ImportError:
    option_menu = None

from model.model_loader import (
    detect_device,
    load_model,
    ARCHITECTURES,
    list_uploaded_models,
    save_uploaded_model,
    delete_uploaded_model,
    infer_arch_key,
)
from utils.helper import card, inject_css
from pages import prediction, explainability, metrics, about

st.set_page_config(
    page_title="Fetal HC AI Platform",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init() -> None:
    inject_css()
    if "device_info" not in st.session_state:
        st.session_state.device_info = detect_device(prefer_cuda=True)
    if "arch_key" not in st.session_state:
        st.session_state.arch_key = "nnUNet_ResAttn"
    if "model_details" not in st.session_state:
        # Auto-load top model from Best_Models if present, else fallback
        avail = list_uploaded_models()
        top_model_path = None
        for p in avail:
            if "best_model_attn" in p.name.lower() or "nnunet_resattn" in p.name.lower():
                top_model_path = p
                break
        if not top_model_path and avail:
            top_model_path = avail[0]
            
        if top_model_path:
            arch = infer_arch_key(top_model_path.name)
            st.session_state.arch_key = arch
            st.session_state.model_details = load_model(
                arch,
                checkpoint_path=top_model_path,
                device=st.session_state.device_info.active
            )
        else:
            st.session_state.model_details = load_model(
                st.session_state.arch_key,
                checkpoint_path=None,
                device=st.session_state.device_info.active
            )

    defaults = {
        "threshold": 0.50,
        "image_size": 384,
        "pixel_spacing": 0.15,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-brand-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"></path>
                        <path d="M12 6v6l4 2"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                </div>
                <div class="sidebar-brand-text">
                    <h2>Fetal HC AI</h2>
                    <span>Ultrasound Intelligence</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption("Ultrasound Segmentation & Biometry Workspace")
        st.markdown("---")

        pages = ["Dashboard", "Prediction", "Explainability", "Metrics", "About"]
        icons = ["dashboard", "browse_activity", "visibility", "bar_chart", "info"]

        # Handle programmatically targeted page navigation and persist current selection
        if "current_page" not in st.session_state:
            st.session_state.current_page = "Dashboard"
            
        if "nav_target" in st.session_state and st.session_state.nav_target in pages:
            st.session_state.current_page = st.session_state.pop("nav_target")

        default_idx = pages.index(st.session_state.current_page)

        if option_menu:
            selected = option_menu(
                menu_title=None,
                options=pages,
                icons=icons,
                default_index=default_idx,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": "#60A5FA", "font-size": "1rem"},
                    "nav-link": {
                        "font-size": "0.9rem",
                        "text-align": "left",
                        "margin": "3px 0",
                        "border-radius": "8px",
                        "color": "#CBD5E1",
                        "font-weight": "500",
                    },
                    "nav-link-selected": {"background-color": "#0066FF", "color": "#FFFFFF", "font-weight": "600"},
                }
            )
            if selected:
                st.session_state.current_page = selected
        else:
            selected = st.radio("Navigation", pages, label_visibility="collapsed", index=default_idx)
            if selected:
                st.session_state.current_page = selected

        st.markdown("---")

        # Hardware Engine Status
        info = st.session_state.device_info
        st.markdown("##### Compute Engine")
        if info.active == "cuda":
            st.markdown(
                f'<div class="badge badge-cuda"><span class="badge-dot"></span> GPU: {info.gpu_name or "CUDA Active"}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="badge badge-cpu"><span class="badge-dot"></span> CPU Mode</div>',
                unsafe_allow_html=True
            )
        st.caption(f"PyTorch {info.torch_version} | CUDA {info.cuda_version or 'N/A'}")

        st.markdown("---")

        # Model Settings & Auto Switcher
        with st.expander("Model Settings & Library", expanded=True, icon=":material/tune:"):
            details = st.session_state.model_details
            existing_files = list_uploaded_models()
            
            if existing_files:
                file_map = {f.name: f for f in existing_files}
                file_names = list(file_map.keys())
                
                active_fn = details.checkpoint_path.name if (details.is_uploaded and details.checkpoint_path) else file_names[0]
                sel_idx = file_names.index(active_fn) if active_fn in file_names else 0
                
                selected_file = st.selectbox(
                    "Active Model Checkpoint",
                    options=file_names,
                    index=sel_idx,
                    key="active_model_picker"
                )
                
                # Instant auto-switch when dropdown selection changes
                if not details.is_uploaded or (details.checkpoint_path and details.checkpoint_path.name != selected_file):
                    target_p = file_map[selected_file]
                    inferred_a = infer_arch_key(selected_file)
                    st.session_state.arch_key = inferred_a
                    st.session_state.model_details = load_model(inferred_a, target_p, info.active)
                    st.rerun()
                    
                st.caption(f"**Architecture:** `{st.session_state.arch_key}` | **Size:** {details.size}")
                st.markdown(f'<div class="badge badge-cuda" style="background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); padding: 0.3rem 0.6rem;"><span class="badge-dot" style="background: #34D399;"></span> Active & Ready</div>', unsafe_allow_html=True)
            else:
                st.warning("No pre-trained model files found in Best_Models.")

            st.markdown("---")

            # Upload New Checkpoint File
            uploaded_file = st.file_uploader(
                "Upload Custom Model (.pth, .pt)",
                type=["pth", "pt", "ckpt", "bin", "onnx"],
                key="manual_model_upload"
            )
            if uploaded_file and st.button("Save & Activate Model", width="stretch", icon=":material/upload:"):
                saved_path = save_uploaded_model(uploaded_file)
                inferred_a = infer_arch_key(uploaded_file.name)
                st.session_state.arch_key = inferred_a
                st.session_state.model_details = load_model(inferred_a, saved_path, info.active)
                st.success(f"Loaded {uploaded_file.name}")
                st.rerun()

        with st.expander("Prediction Parameters", expanded=False, icon=":material/settings:"):
            st.session_state.threshold = st.slider("Confidence threshold", 0.10, 1.00, st.session_state.threshold, 0.05)
            st.session_state.image_size = st.selectbox("Resolution (px)", [256, 384, 512], index=[256, 384, 512].index(st.session_state.image_size))
            st.session_state.pixel_spacing = st.number_input("Pixel Spacing (mm/px)", 0.05, 0.50, st.session_state.pixel_spacing, 0.01)

        if st.button("Reset Workspace", width="stretch", icon=":material/restart_alt:"):
            for key in ["original", "truth", "result"]:
                st.session_state.pop(key, None)
            st.rerun()

    return selected

def dashboard() -> None:
    # Clean Hero Banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(0,102,255,0.12) 0%, rgba(0,210,255,0.04) 100%); border: 1px solid rgba(0,102,255,0.2); border-radius: 16px; padding: 1.75rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <span style="background: rgba(0,102,255,0.2); color: #60A5FA; font-size: 0.75rem; font-weight: 700; padding: 0.25rem 0.65rem; border-radius: 9999px; text-transform: uppercase; letter-spacing: 0.08em;">Fetal Ultrasound Workspace</span>
                    <h1 style="margin: 0.5rem 0 0.25rem 0; font-size: 2rem; color: #F8FAFC;">Fetal Head Circumference AI Platform</h1>
                    <p style="margin: 0; color: #94A3B8; font-size: 0.95rem;">AI-assisted segmentation, automated ellipse fitting, and fetal head biometry estimation.</p>
                </div>
                <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 0.75rem 1.25rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">System Status</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #34D399; font-family: 'Outfit'; margin-top: 2px;">Ready for Testing</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info = st.session_state.device_info
    details = st.session_state.model_details
    has_result = "result" in st.session_state

    # Overview Cards
    st.markdown("#### Workspace Status")
    cols = st.columns(4)
    with cols[0]:
        model_display = details.checkpoint_path.name if (details.is_uploaded and details.checkpoint_path) else "No Model Uploaded"
        sub_desc = f"{details.arch_key} | {details.size}" if details.is_uploaded else "Upload .pth file in sidebar"
        card("Active Model", model_display, sub_desc, style="accent" if details.is_uploaded else "warning")
    with cols[1]:
        card("Compute Engine", info.label, info.gpu_name or info.reason, style="success" if info.active == "cuda" else "warning")
    with cols[2]:
        card("Input Resolution", f"{st.session_state.image_size} px", f"Spacing: {st.session_state.pixel_spacing} mm/px", style="default")
    with cols[3]:
        card("Prediction Status", "Completed" if has_result else "Awaiting Input", "Results available" if has_result else "No image loaded yet", style="success" if has_result else "default")

    st.markdown("<br>", unsafe_allow_html=True)

    # Start Workspace Section
    with st.container(border=True):
        st.subheader("Start Ultrasound Analysis", divider="blue")
        st.write("Upload an ultrasound image or select a sample frame in the **Prediction** tab to run segmentation and view fetal head measurements.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown(
                """
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1rem;">
                    <div style="color: #60A5FA; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.25rem;">1. Select Image</div>
                    <div style="color: #94A3B8; font-size: 0.82rem;">Upload a 2D ultrasound scan or choose a test sample frame.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                """
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1rem;">
                    <div style="color: #60A5FA; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.25rem;">2. Run AI Segmentation</div>
                    <div style="color: #94A3B8; font-size: 0.82rem;">Execute deep learning inference & ellipse contour fitting.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                """
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 1rem;">
                    <div style="color: #60A5FA; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.25rem;">3. Clinical Metrics</div>
                    <div style="color: #94A3B8; font-size: 0.82rem;">Review HC (mm), Gestational Age, and Grad-CAM++ saliency.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Open Prediction Workspace", type="primary", width="stretch", icon=":material/play_circle:"):
            st.session_state.nav_target = "Prediction"
            st.rerun()

init()
active = sidebar()

if active == "Dashboard":
    dashboard()
elif active == "Prediction":
    prediction.render()
elif active == "Explainability":
    explainability.render()
elif active == "Metrics":
    metrics.render()
else:
    about.render()
