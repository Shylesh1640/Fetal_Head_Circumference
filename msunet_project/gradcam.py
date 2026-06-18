"""
gradcam.py
----------
Grad-CAM for the segmentation network, used for interpretability as
described in the paper ("Grad-CAM visualizations further enhance clinical
interpretability ... highlighting regions that constitute the influences
on model predictions").

For segmentation models there is no single scalar class score, so we
follow the standard adaptation used in segmentation Grad-CAM work: the
"score" being explained is the mean predicted (sigmoid) probability over
all foreground pixels, and gradients are captured at a chosen encoder
feature map (by default, the deepest MiT-B2 stage feeding the bottleneck).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SegmentationGradCAM:
    """
    Usage
    -----
        # Target a layer that outputs a plain Tensor (NOT a list/tuple).
        # For the MiT-B2 encoder used in this project, `patch_embed4`
        # (the deepest stage's patch-embedding conv) is the recommended
        # target: it is the last point in the encoder where the feature
        # map is a single [B, C, h, w] tensor before being split into the
        # multi-stage list returned by `encoder.forward`.
        target_layer = model.encoder.patch_embed4
        cam_extractor = SegmentationGradCAM(model, target_layer=target_layer)
        heatmap = cam_extractor(input_tensor)   # [H, W] numpy array in [0, 1]
        cam_extractor.remove_hooks()
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        # MiT-B2 encoder (and most SMP encoders) return a LIST of feature
        # maps for each stage when called as the full encoder; if a single
        # tensor is returned instead, handle that too.
        feat = out[-1] if isinstance(out, (list, tuple)) else out
        self.activations = feat

    def _save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[-1] if isinstance(grad_output, (list, tuple)) else grad_output[0]
        self.gradients = grad

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, upsample_size=None) -> np.ndarray:
        """
        input_tensor : [1, 3, H, W], requires_grad not necessary (we only
                        need gradients w.r.t. the target layer's activations).
        upsample_size : (H, W) to resize the CAM to; defaults to the input
                        tensor's spatial size.
        """
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(False)

        logits = self.model(input_tensor)         # [1, 1, H, W]
        probs = torch.sigmoid(logits)
        score = probs.mean()                       # scalar "explanation target"

        score.backward(retain_graph=False)

        activations = self.activations             # [1, C, h, w]
        gradients = self.gradients                  # [1, C, h, w]

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        if upsample_size is None:
            upsample_size = input_tensor.shape[-2:]
        cam = F.interpolate(cam, size=upsample_size, mode="bilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam


def overlay_heatmap_on_image(image_rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    image_rgb_uint8 : [H, W, 3] uint8 RGB image
    cam             : [H, W] float array in [0, 1]
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * image_rgb_uint8).astype(np.uint8)
    return overlay


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import model as model_module

    net = model_module.MSUNet(encoder_weights=None)
    net.eval()

    # patch_embed4 is the deepest MiT-B2 stage that outputs a plain Tensor
    # (the encoder's top-level forward returns a List[Tensor], which
    # backward hooks cannot attach to directly).
    target_layer = net.encoder.patch_embed4
    cam_extractor = SegmentationGradCAM(net, target_layer=target_layer)

    x = torch.randn(1, 3, 256, 256)
    heatmap = cam_extractor(x)
    print("CAM shape:", heatmap.shape, "min/max:", heatmap.min(), heatmap.max())
    assert heatmap.shape == (256, 256)

    fake_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    overlay = overlay_heatmap_on_image(fake_rgb, heatmap)
    print("Overlay shape:", overlay.shape, overlay.dtype)
    assert overlay.shape == (256, 256, 3)

    cam_extractor.remove_hooks()
    print("OK")
