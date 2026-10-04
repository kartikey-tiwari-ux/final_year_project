"""Grad-CAM (Selvaraju et al. 2017) for the image branch — implemented directly via
forward/backward hooks, no extra library. See PROJECT_ARCHITECTURE.md.

Works against any torch conv backbone + a linear scoring vector for the predicted
class. For Model B2 (frozen ResNet18 + sklearn LogisticRegression), the sklearn
classifier's coef_ row for the predicted class is used as that linear scoring vector,
letting Grad-CAM trace gradients back through ResNet18's conv layers even though the
final classification step itself lives outside PyTorch.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_weight_vector: torch.Tensor) -> np.ndarray:
        """image_tensor: (1, 3, H, W). class_weight_vector: (conv_out_channels,) —
        the linear scoring vector for the target class, applied after global average
        pooling of the backbone's final feature map (matches how both the CNN
        classifier and the frozen-ResNet+LogReg pipeline actually score an image)."""
        image_tensor = image_tensor.clone().requires_grad_(True)
        features = self.model(image_tensor)  # forward triggers the hook -> self.activations

        pooled = features.mean(dim=[2, 3]) if features.dim() == 4 else features
        score = (pooled.squeeze(0) * class_weight_vector).sum()

        self.model.zero_grad()
        score.backward()

        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # global-average-pool the gradients
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (1, 1, H, W)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
        return cam
