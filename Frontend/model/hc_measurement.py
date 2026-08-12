"""Contour and ellipse-based fetal head circumference & gestational age estimation."""
from dataclasses import dataclass
import math
import cv2
import numpy as np

@dataclass
class HCMeasurement:
    circumference_mm: float | None
    area_px: int
    gestational_age_weeks: float | None
    gestational_age_str: str
    major_axis_mm: float = 0.0
    minor_axis_mm: float = 0.0
    bpd_mm: float = 0.0  # Biparietal Diameter
    ofd_mm: float = 0.0  # Occipitofrontal Diameter
    visualization: np.ndarray = None

def estimate_head_circumference(image_rgb: np.ndarray, mask: np.ndarray, pixel_spacing_mm: float = 0.15) -> HCMeasurement:
    """Fit largest contour, calculate HC (Ramanujan approx) and Hadlock Gestational Age."""
    canvas = image_rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return HCMeasurement(None, 0, None, "N/A", visualization=canvas)
        
    contour = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(contour))
    
    if len(contour) < 5 or area < 100:
        return HCMeasurement(None, area, None, "N/A", visualization=canvas)
        
    # Draw segmentation contour (Cyan)
    cv2.drawContours(canvas, [contour], -1, (0, 210, 255), 2)
    
    # Fit ellipse
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    
    # Fitted ellipse line (Vibrant Blue)
    cv2.ellipse(canvas, ((cx, cy), (axis_a, axis_b), angle), (0, 102, 255), 2)
    
    # Center crosshair
    cv2.drawMarker(canvas, (int(cx), int(cy)), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    
    major_px, minor_px = max(axis_a, axis_b), min(axis_a, axis_b)
    major_mm = major_px * pixel_spacing_mm
    minor_mm = minor_px * pixel_spacing_mm
    
    # Semi-axes in mm
    a, b = major_mm / 2.0, minor_mm / 2.0
    
    # Ramanujan's 2nd approximation for ellipse perimeter
    h = ((a - b) ** 2) / ((a + b) ** 2)
    hc_mm = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
    
    # Hadlock Formula for Gestational Age (GA in weeks)
    # log10(GA) = 1.1143 + 0.00624 * HC - 0.00000684 * (HC^2)
    try:
        log_ga = 1.1143 + (0.00624 * hc_mm) - (0.00000684 * (hc_mm ** 2))
        ga_weeks_float = 10 ** log_ga
        w = int(ga_weeks_float)
        d = int(round((ga_weeks_float - w) * 7))
        if d == 7:
            w += 1
            d = 0
        ga_str = f"{w}w {d}d"
    except Exception:
        ga_weeks_float = None
        ga_str = "N/A"

    # Annotate on canvas
    text = f"HC: {hc_mm:.1f} mm | GA: {ga_str}"
    text_pos = (max(10, int(cx) - 100), max(30, int(cy - major_px / 2) - 15))
    
    # Text background pill
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(canvas, (text_pos[0] - 4, text_pos[1] - th - 4), (text_pos[0] + tw + 4, text_pos[1] + baseline), (11, 15, 25), -1)
    cv2.putText(canvas, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)

    return HCMeasurement(
        circumference_mm=hc_mm,
        area_px=area,
        gestational_age_weeks=ga_weeks_float,
        gestational_age_str=ga_str,
        major_axis_mm=major_mm,
        minor_axis_mm=minor_mm,
        bpd_mm=minor_mm,
        ofd_mm=major_mm,
        visualization=canvas,
    )
