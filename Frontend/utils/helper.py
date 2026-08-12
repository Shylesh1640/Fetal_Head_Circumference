"""Shared display, custom CSS injection, and report export helpers."""
from __future__ import annotations
import io
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st


def inject_css() -> None:
    """Inject modern, elegant dark glassmorphic CSS styling for medical AI dashboard."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

        /* Root style resets & fonts */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        h1, h2, h3, .main-header-title {
            font-family: 'Outfit', 'Inter', sans-serif !important;
            letter-spacing: -0.02em;
        }

        /* Material Symbols Outlined Font Definition */
        .material-symbols-outlined {
            font-family: 'Material Symbols Outlined' !important;
            font-weight: normal;
            font-style: normal;
            font-size: 20px;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            display: inline-block;
            white-space: nowrap;
            word-wrap: normal;
            direction: ltr;
            -webkit-font-smoothing: antialiased;
        }

        /* App Background & Canvas */
        .stApp {
            background-color: #0B0F19;
            background-image: 
                radial-gradient(at 0% 0%, rgba(0, 102, 255, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(0, 210, 255, 0.05) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(15, 23, 42, 0.6) 0px, transparent 50%);
            background-attachment: fixed;
        }

        /* Glassmorphic Containers */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(21, 28, 44, 0.65) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25) !important;
            padding: 1.25rem !important;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(0, 102, 255, 0.3) !important;
            box-shadow: 0 10px 40px 0 rgba(0, 102, 255, 0.12) !important;
        }

        /* Custom Metric Cards */
        .kpi-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 4px solid #0066FF;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.5rem;
            position: relative;
            overflow: hidden;
        }

        .kpi-card-accent {
            border-left-color: #00D2FF;
        }

        .kpi-card-success {
            border-left-color: #10B981;
        }

        .kpi-card-warning {
            border-left-color: #F59E0B;
        }

        .kpi-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }

        .kpi-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #F8FAFC;
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }

        .kpi-note {
            font-size: 0.75rem;
            color: #64748B;
            margin-top: 0.35rem;
        }

        /* Buttons Styling */
        .stButton button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #0066FF 0%, #0052CC 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            box-shadow: 0 4px 14px 0 rgba(0, 102, 255, 0.35) !important;
        }

        .stButton button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px 0 rgba(0, 102, 255, 0.5) !important;
            background: linear-gradient(135deg, #1A75FF 0%, #0066FF 100%) !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #080C14 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        section[data-testid="stSidebar"] .stMarkdown h1 {
            font-size: 1.4rem !important;
            color: #F8FAFC !important;
        }

        /* Sidebar Header Container Fix */
        .sidebar-brand-box {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
            padding: 0.25rem 0;
        }

        .sidebar-brand-icon {
            background: linear-gradient(135deg, #0066FF, #00D2FF);
            width: 38px;
            height: 38px;
            min-width: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            box-shadow: 0 4px 12px rgba(0,102,255,0.35);
        }

        .sidebar-brand-text h2 {
            margin: 0 !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
            line-height: 1.2 !important;
        }

        .sidebar-brand-text span {
            font-size: 0.68rem !important;
            color: #94A3B8 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            display: block !important;
        }

        /* Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.75rem;
            border-radius: 8px;
            font-size: 0.78rem;
            font-weight: 600;
            gap: 0.4rem;
            line-height: 1.3;
            max-width: 100%;
            box-sizing: border-box;
        }

        .badge-dot {
            width: 8px;
            height: 8px;
            min-width: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        .badge-cuda {
            background: rgba(16, 185, 129, 0.12);
            color: #34D399;
            border: 1px solid rgba(52, 211, 153, 0.25);
        }
        .badge-cuda .badge-dot {
            background: #34D399;
            box-shadow: 0 0 8px #34D399;
        }

        .badge-cpu {
            background: rgba(245, 158, 11, 0.12);
            color: #FBBF24;
            border: 1px solid rgba(251, 191, 36, 0.25);
        }
        .badge-cpu .badge-dot {
            background: #FBBF24;
            box-shadow: 0 0 8px #FBBF24;
        }

        /* Visual Dividers */
        hr {
            border-color: rgba(255, 255, 255, 0.08) !important;
            margin: 1rem 0 !important;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.6);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 8px;
            font-weight: 500;
            color: #94A3B8;
            padding: 0 16px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #0066FF !important;
            color: #FFFFFF !important;
            font-weight: 600;
        }

        /* Error Analysis Legend & Card Styling */
        .error-legend-box {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 0.85rem 1.1rem;
            margin-top: 0.75rem;
            margin-bottom: 0.5rem;
        }

        .legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.75rem;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 0.5rem 0.65rem;
        }

        .legend-swatch {
            width: 14px;
            height: 14px;
            min-width: 14px;
            border-radius: 4px;
            box-shadow: 0 0 6px rgba(0, 0, 0, 0.4);
        }

        .legend-info {
            display: flex;
            flex-direction: column;
            line-height: 1.2;
        }

        .legend-title {
            font-size: 0.76rem;
            font-weight: 600;
            color: #F8FAFC;
        }

        .legend-detail {
            font-size: 0.68rem;
            color: #94A3B8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(label: str, value: str, note: str = "", style: str = "default") -> None:
    """Render a modern glassmorphic KPI card."""
    accent_class = ""
    if style == "accent":
        accent_class = "kpi-card-accent"
    elif style == "success":
        accent_class = "kpi-card-success"
    elif style == "warning":
        accent_class = "kpi-card-warning"

    st.markdown(
        f"""
        <div class="kpi-card {accent_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {f'<div class="kpi-note">{note}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_image(title: str, image, caption: str = "") -> None:
    """Render an image inside a consistent glassmorphic bordered section."""
    with st.container(border=True):
        st.subheader(title, divider="blue")
        st.image(image, caption=caption, width="stretch")


def metric_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1.0 else f"{value:.2f}"
    return str(value)


def report_csv(metrics: dict[str, object], model_name: str) -> bytes:
    row = {
        "Prediction Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model Used": model_name,
        **metrics,
    }
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def make_pdf(metrics: dict[str, object], model_name: str) -> bytes | None:
    """Generate a clean clinical PDF report if reportlab is installed."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)

        # Header Banner
        pdf.setFillColor(colors.HexColor("#0B0F19"))
        pdf.rect(0, 720, 612, 72, fill=True, stroke=False)

        pdf.setFillColor(colors.HexColor("#0066FF"))
        pdf.rect(0, 716, 612, 4, fill=True, stroke=False)

        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(40, 752, "Fetal Head Circumference AI Report")

        pdf.setFont("Helvetica", 9)
        pdf.setFillColor(colors.HexColor("#94A3B8"))
        pdf.drawString(40, 735, f"Clinical Research & Quantitative Analysis Platform | Generated: {datetime.now():%d %b %Y, %H:%M}")

        # Metadata Section
        pdf.setFillColor(colors.HexColor("#1E293B"))
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, 685, "System Configuration & Model Metadata")

        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.black)
        pdf.drawString(50, 665, f"Active Architecture: {model_name}")
        pdf.drawString(50, 650, "Analysis Status: Completed Successfully")

        # Table Header
        pdf.setFillColor(colors.HexColor("#F1F5F9"))
        pdf.rect(40, 615, 532, 22, fill=True, stroke=False)
        pdf.setFillColor(colors.HexColor("#0F172A"))
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(50, 622, "Evaluation Metric")
        pdf.drawString(300, 622, "Quantitative Value")

        y = 595
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#334155"))

        for idx, (key, value) in enumerate(metrics.items()):
            if idx % 2 == 1:
                pdf.setFillColor(colors.HexColor("#F8FAFC"))
                pdf.rect(40, y - 4, 532, 18, fill=True, stroke=False)
                pdf.setFillColor(colors.HexColor("#334155"))

            pdf.drawString(50, y, str(key))
            pdf.drawString(300, y, metric_value(value))
            y -= 20

            if y < 80:
                pdf.showPage()
                y = 730

        # Footer
        pdf.setFont("Helvetica-Oblique", 8)
        pdf.setFillColor(colors.HexColor("#64748B"))
        pdf.drawString(40, 30, "Disclaimer: For research and clinical decision support only. Validate against medical guidelines.")

        pdf.save()
        return buffer.getvalue()
    except Exception:
        return None
