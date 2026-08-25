"""
plot_roc_curves.py

Generates a real One-vs-Rest ROC curve plot (per-class AUC), styled like the
reference image, from your ACTUAL trained model(s) run against your ACTUAL
labeled test set.

This is NOT a mockup / fake-data script -- it does real inference and real
sklearn ROC computation, exactly like evaluate.py does for the confusion
matrix. It just wasn't in the original project, so predictions never got
turned into an ROC plot.

REQUIREMENTS TO RUN THIS FOR REAL (none of these exist in the sandbox that
wrote this file -- it was written blind, without running it, because torch
isn't installed there and there's no populated data/test/ folder there):
  1. torch + torchvision installed (pip install -r requirements.txt)
  2. A populated ./data/test/<class_folder>/*.jpg for all 8 classes (this is
     the same folder evaluate.py needs -- see fill_ak_folder.py / README.md
     for how to build it from the raw ISIC archive)
  3. The three .pth weight files in this same folder

USAGE:
    # ROC for a single model:
    python plot_roc_curves.py --model efficientnet-b3

    # ROC for the calibrated 3-model ensemble (recommended -- matches what
    # patients actually see in the app):
    python plot_roc_curves.py --model ensemble

Output: static/roc_curve_<model>.png
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models import load_model

# --- CONFIGURATION (must exactly match app.py / evaluate.py) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Actinic Keratosis",        # 0 - AK
    "Basal Cell Carcinoma",     # 1 - BCC
    "Benign Keratosis",         # 2 - BKL
    "Dermatofibroma",           # 3 - DF
    "Melanoma",                 # 4 - MEL
    "Melanocytic Nevus",        # 5 - NV
    "Squamous Cell Carcinoma",  # 6 - SCC
    "Vascular Lesion",          # 7 - VASC
]
NUM_CLASSES = len(CLASS_NAMES)
TEST_DIR = "./data/test"
STATIC_DIR = "./static"

MODEL_WEIGHTS = {
    "resnet50": "resnet50_best.pth",
    "efficientnet-b3": "efficientnet-b3_best.pth",
    "visiontransformer": "visiontransformer_best.pth",
}

# Distinct, readable colors matching the reference plot's palette style
CLASS_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#006400", "#f781bf",
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_probs_single_model(model_name, loader):
    weights_file = MODEL_WEIGHTS[model_name]
    model = load_model(model_name, NUM_CLASSES, weights_file, DEVICE)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def get_probs_ensemble(loader, temperature=1.5):
    models_dict = {
        name: load_model(name, NUM_CLASSES, wf, DEVICE)
        for name, wf in MODEL_WEIGHTS.items()
    }

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            batch_probs = torch.zeros(images.size(0), NUM_CLASSES, device=DEVICE)
            for model in models_dict.values():
                logits = model(images) / temperature
                batch_probs += F.softmax(logits, dim=1)
            batch_probs /= len(models_dict)
            all_probs.append(batch_probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def plot_roc(y_true, y_probs, model_label):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(9, 8))
    for i, cname in enumerate(CLASS_NAMES):
        if y_true_bin[:, i].sum() == 0:
            print(f"  Skipping {cname}: no positive samples in this test set")
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=CLASS_COLORS[i % len(CLASS_COLORS)], lw=2,
                 label=f"{cname} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    os.makedirs(STATIC_DIR, exist_ok=True)
    out_path = os.path.join(STATIC_DIR, f"roc_curve_{model_label}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ROC curves from real model inference.")
    parser.add_argument("--model", default="ensemble",
                         choices=list(MODEL_WEIGHTS.keys()) + ["ensemble"],
                         help="Which model to evaluate (default: ensemble of all 3).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(TEST_DIR):
        print(f"ERROR: '{TEST_DIR}' not found. This script needs the same "
              f"data/test/<class_folder>/*.jpg structure evaluate.py uses.")
        return

    dataset = ImageFolder(root=TEST_DIR, transform=transform)
    if len(dataset.classes) != NUM_CLASSES:
        print(f"ERROR: found {len(dataset.classes)} class folders "
              f"({dataset.classes}) but CLASS_NAMES has {NUM_CLASSES} entries. "
              f"These must match exactly (see evaluate.py's verify_class_order).")
        return
    print(f"Detected folder order: {dataset.classes}")
    print(f"Configured CLASS_NAMES: {CLASS_NAMES}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=torch.cuda.is_available())

    print(f"\nRunning inference with: {args.model}  (device={DEVICE})")
    if args.model == "ensemble":
        y_probs, y_true = get_probs_ensemble(loader)
    else:
        y_probs, y_true = get_probs_single_model(args.model, loader)

    plot_roc(y_true, y_probs, args.model)


if __name__ == "__main__":
    main()
