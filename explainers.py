"""
explainers.py
Real (non-mocked) implementations of Grad-CAM, LIME, and SHAP for the
ResNet-50 / EfficientNet-B3 / ViT-B/16 skin lesion classifiers.

This file was missing from the uploaded project (app.py imports it but it
didn't exist), which is why Grad-CAM/LIME/SHAP were never actually running.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from lime import lime_image
from skimage.segmentation import mark_boundaries

import shap

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# --------------------------------------------------------------------------
# Grad-CAM (works for CNNs directly; for ViT it reshapes patch tokens back
# into a 2D grid, dropping the CLS token, which is the standard approach for
# transformer Grad-CAM / "Attention rollout"-style visualizations).
# --------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        if activations.dim() == 3:
            # ViT: activations/gradients are (batch, tokens, hidden_dim).
            # Token 0 is the CLS token - drop it and reshape the remaining
            # patch tokens into a square spatial grid.
            acts = activations[0, 1:, :]
            grads = gradients[0, 1:, :]
            weights = grads.mean(dim=0)
            cam = torch.matmul(acts, weights)
            num_patches = cam.shape[0]
            grid_size = int(num_patches ** 0.5)
            cam = cam[: grid_size * grid_size].reshape(grid_size, grid_size)
        else:
            # CNN: activations/gradients are (batch, C, H, W)
            acts = activations[0]
            grads = gradients[0]
            weights = grads.mean(dim=(1, 2))
            cam = torch.einsum('c,chw->hw', weights, acts)

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        return cam


def _tensor_to_uint8_rgb(img_tensor):
    """Undo ImageNet normalization -> uint8 HWC RGB image (224,224,3)."""
    img = img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _make_predict_fn(model, device):
    def predict(images_uint8):
        model.eval()
        batch = []
        for im in images_uint8:
            norm = (im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.from_numpy(norm.transpose(2, 0, 1)).float()
            batch.append(t)
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()
    return predict


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------
def get_lime_explanation(model, img_tensor, transform, device, num_samples=150):
    rgb_img = _tensor_to_uint8_rgb(img_tensor)
    predict_fn = _make_predict_fn(model, device)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        rgb_img,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=8, hide_rest=False
    )
    overlay = mark_boundaries(temp / 255.0, mask)
    return (overlay * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# SHAP (model-agnostic PartitionExplainer over image patches — this works
# identically for CNNs and ViT because it only calls the predict function,
# it never needs internal gradients or layer access)
#
# NOTE on speed: max_evals is the number of masked-image forward passes SHAP
# runs to estimate attributions. Lowered from 100 -> 60 by request: this is
# roughly 40% fewer forward passes (meaningfully faster, especially on CPU
# and especially for the heavier ViT model), at the cost of a slightly
# coarser/noisier heatmap. If you want the original higher-fidelity output
# back, just change max_evals=60 back to max_evals=100 below.
# --------------------------------------------------------------------------
def get_shap_explanation(model, img_tensor, class_idx, device, max_evals=60):
    rgb_img = _tensor_to_uint8_rgb(img_tensor)
    predict_fn = _make_predict_fn(model, device)

    masker = shap.maskers.Image("blur(32,32)", rgb_img.shape)
    explainer = shap.Explainer(predict_fn, masker)

    shap_values = explainer(
        np.expand_dims(rgb_img, axis=0),
        max_evals=max_evals,
        batch_size=32,
        outputs=[class_idx],
    )
    # shap_values.values shape: (1, H, W, C, 1) for the single requested output
    sv = shap_values.values[0, ..., 0]
    heat = np.abs(sv).sum(axis=-1)
    return heat