"""Model loading, device detection, and architecture registry."""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure project paths are in sys.path
FRONTEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = FRONTEND_DIR.parent
for p in [str(FRONTEND_DIR), str(ROOT_DIR), str(ROOT_DIR / "Experiments")]:
    if p not in sys.path:
        sys.path.insert(0, p)

@dataclass
class DeviceInfo:
    """Runtime accelerator capabilities exposed in the UI."""
    requested: str
    active: str
    label: str
    gpu_name: str | None = None
    torch_version: str = "Unavailable"
    cuda_version: str | None = None
    reason: str = ""

@dataclass
class ModelDetails:
    name: str = "No Model Uploaded"
    arch_key: str = "nnUNet_ResAttn"
    parameters: str = "N/A"
    size: str = "N/A"
    device: str = "CPU"
    dice_score: float = 0.0
    hd95: float = 0.0
    description: str = "No manual model checkpoint loaded yet."
    model: Any | None = None
    is_uploaded: bool = False
    checkpoint_path: Path | None = None
    status: str = "No Model Loaded"

ARCHITECTURES = {
    "nnUNet_ResAttn": {
        "name": "nnUNet_ResAttn",
        "label": "nnUNet_ResAttn (Top Performer)",
        "params": "14.8 M",
        "size": "59.2 MB",
        "dice": 0.9643,
        "hd95": "16.7 mm",
        "desc": "nnU-Net backbone + Residual blocks + Attention Gates on skip connections.",
    },
    "nnUNet_MSFFBridge_ResAttn": {
        "name": "nnUNet_MSFFBridge_ResAttn",
        "label": "nnUNet_MSFFBridge_ResAttn (Top Performer)",
        "params": "16.2 M",
        "size": "64.8 MB",
        "dice": 0.9638,
        "hd95": "17.1 mm",
        "desc": "nnUNet + Residual Attention + Multi-Scale Feature Fusion (ASPP) Bridge at bottleneck.",
    },
    "nnUNet": {
        "name": "nnUNet",
        "label": "nnUNet (Standard Baseline)",
        "params": "14.2 M",
        "size": "56.8 MB",
        "dice": 0.9582,
        "hd95": "19.2 mm",
        "desc": "Practical 2D adaptation of self-configuring nnU-Net plain-block design.",
    },
    "ResAttnUNet": {
        "name": "ResAttnUNet",
        "label": "ResAttnUNet",
        "params": "14.5 M",
        "size": "58.0 MB",
        "dice": 0.9241,
        "hd95": "24.5 mm",
        "desc": "Residual U-Net encoder/decoder with Oktay attention gates.",
    },
    "UNet_Hybrid": {
        "name": "UNet_Hybrid",
        "label": "UNet_Hybrid (TransUNet style)",
        "params": "18.5 M",
        "size": "74.0 MB",
        "dice": 0.9125,
        "hd95": "28.3 mm",
        "desc": "CNN Encoder with Transformer Self-Attention Bottleneck (TransUNet style).",
    },
    "UNet_MiTB2": {
        "name": "UNet_MiTB2",
        "label": "UNet_MiTB2 (SegFormer)",
        "params": "24.7 M",
        "size": "98.8 MB",
        "dice": 0.9054,
        "hd95": "31.2 mm",
        "desc": "SegFormer Mix Transformer (MiT-B2) encoder backbone.",
    },
    "msunet": {
        "name": "msunet",
        "label": "msunet (Multi-Scale U-Net)",
        "params": "13.9 M",
        "size": "55.6 MB",
        "dice": 0.8872,
        "hd95": "35.8 mm",
        "desc": "Multi-Scale U-Net fed with an image pyramid downsampled at each stage.",
    },
}

def get_uploaded_models_dir() -> Path:
    """Return directory path dedicated to storing manually uploaded model files."""
    upload_dir = FRONTEND_DIR / "uploaded_models"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir

def get_best_models_dir() -> Path:
    """Return directory containing pre-trained Best_Models checkpoints."""
    return ROOT_DIR / "Best_Models"

def list_uploaded_models() -> list[Path]:
    """List all available model files from uploaded_models/ and Best_Models/."""
    upload_dir = get_uploaded_models_dir()
    best_dir = get_best_models_dir()
    allowed_exts = {".pth", ".pt", ".ckpt", ".bin", ".onnx"}
    
    files = []
    seen = set()
    for d in [upload_dir, best_dir]:
        if d.exists():
            for p in d.glob("*"):
                if p.suffix.lower() in allowed_exts and p.is_file() and p.name not in seen:
                    files.append(p)
                    seen.add(p.name)
    return files

def infer_arch_key(filename: str) -> str:
    """Infer matching network architecture key from model filename."""
    fn = filename.lower()
    if "msffbridge" in fn or "dafar_net" in fn or "dafar" in fn:
        return "nnUNet_MSFFBridge_ResAttn"
    elif "best_model_attn" in fn or ("resattn" in fn and "nnunet" in fn):
        return "nnUNet_ResAttn"
    elif "resattn" in fn:
        return "ResAttnUNet"
    elif "nnunet" in fn:
        return "nnUNet"
    elif "msunet" in fn:
        return "msunet"
    elif "mitb2" in fn or "segformer" in fn:
        return "UNet_MiTB2"
    elif "hybrid" in fn:
        return "UNet_Hybrid"
    return "nnUNet_ResAttn"

def save_uploaded_model(uploaded_file) -> Path:
    """Save an uploaded file buffer to the uploaded models directory."""
    upload_dir = get_uploaded_models_dir()
    target_path = upload_dir / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path

def delete_uploaded_model(filename: str) -> bool:
    """Delete a manually uploaded model file by name."""
    upload_dir = get_uploaded_models_dir()
    target_path = upload_dir / filename
    if target_path.exists():
        target_path.unlink()
        return True
    return False

def detect_device(prefer_cuda: bool = True) -> DeviceInfo:
    """Select CUDA when the active PyTorch installation can access it."""
    try:
        import torch
        cuda_ready = torch.cuda.is_available()
        if prefer_cuda and cuda_ready:
            name = torch.cuda.get_device_name(0)
            return DeviceInfo("CUDA", "cuda", "CUDA enabled", name, torch.__version__, torch.version.cuda)
        reason = "CUDA is unavailable in active runtime (CPU mode)."
        if torch.version.cuda is None:
            reason = "Active PyTorch build is CPU-only."
        elif prefer_cuda:
            reason = "CUDA driver/device mismatch."
        return DeviceInfo("CUDA" if prefer_cuda else "CPU", "cpu", "CPU", None, torch.__version__, torch.version.cuda, reason)
    except Exception as exc:
        return DeviceInfo("CUDA" if prefer_cuda else "CPU", "cpu", "CPU", None, "Unavailable", None, f"PyTorch state: {exc}")

def load_model(arch_key: str = "nnUNet_ResAttn", checkpoint_path: str | Path | None = None, device: str = "cpu") -> ModelDetails:
    """Load model ONLY when a manual checkpoint_path is explicitly provided.
    
    If checkpoint_path is None, returns ModelDetails indicating no model is loaded.
    """
    arch_info = ARCHITECTURES.get(arch_key, ARCHITECTURES["nnUNet_ResAttn"])
    
    if not checkpoint_path:
        return ModelDetails(
            name="No Model Uploaded",
            arch_key=arch_key,
            parameters=arch_info["params"],
            size="N/A",
            device=device.upper(),
            dice_score=arch_info["dice"],
            description=arch_info["desc"],
            model=None,
            is_uploaded=False,
            checkpoint_path=None,
            status="Awaiting Manual Model Upload",
        )
    
    path = Path(checkpoint_path)
    if not path.exists():
        return ModelDetails(
            name=f"File Not Found ({path.name})",
            arch_key=arch_key,
            parameters=arch_info["params"],
            size="N/A",
            device=device.upper(),
            dice_score=arch_info["dice"],
            description="Selected checkpoint file does not exist.",
            model=None,
            is_uploaded=False,
            checkpoint_path=None,
            status="Error: Checkpoint File Not Found",
        )
    
    size_mb = path.stat().st_size / (1024 * 1024)
    size_str = f"{size_mb:.2f} MB"
    model_obj = None
    status = "Loaded Successfully"

    try:
        import torch
        import models_library
        if arch_key in models_library.MODEL_REGISTRY:
            builder = models_library.MODEL_REGISTRY[arch_key]
            model_obj = builder()
            
            # Load PyTorch state dict safely
            try:
                state_dict = torch.load(path, map_location=device.lower())
                if isinstance(state_dict, dict):
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    elif "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    elif "model" in state_dict and isinstance(state_dict["model"], dict):
                        state_dict = state_dict["model"]
                model_obj.load_state_dict(state_dict, strict=False)
            except Exception as load_exc:
                status = f"Loaded architecture (weights warning: {load_exc})"
            
            if device.lower() == "cuda" and torch.cuda.is_available():
                model_obj = model_obj.cuda()
            model_obj.eval()
    except Exception as exc:
        status = f"Error instantiating model: {exc}"

    return ModelDetails(
        name=f"{arch_info['label']} [{path.name}]",
        arch_key=arch_key,
        parameters=arch_info["params"],
        size=size_str,
        device=device.upper(),
        dice_score=arch_info["dice"],
        description=arch_info["desc"],
        model=model_obj,
        is_uploaded=True,
        checkpoint_path=path,
        status=status,
    )

