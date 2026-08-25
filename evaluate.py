import argparse
import time
import random
from collections import Counter, defaultdict

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import numpy as np

from models import load_model

try:
    from tqdm import tqdm
except ImportError:
    # Fallback no-op progress bar if tqdm isn't installed, so the script
    # never crashes just because of this optional dependency.
    def tqdm(iterable, **kwargs):
        return iterable

# --- CONFIGURATION (must exactly match app.py) ---
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
TEST_DIR = "./data/test"  # expects data/test/<class_folder>/images...
STATIC_DIR = "./static"
EPOCHS_TRAINED = 15  # update this if you trained for a different number of epochs

MODEL_WEIGHTS = {
    "resnet50": "resnet50_best.pth",
    "efficientnet-b3": "efficientnet-b3_best.pth",
    "visiontransformer": "visiontransformer_best.pth",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def verify_class_order(dataset):
    """
    CRITICAL: PyTorch's ImageFolder assigns label indices 0..N-1 in
    alphabetical order of your subfolder names. If that order doesn't match
    CLASS_NAMES above, every prediction will point at the wrong disease name
    even though the model itself is accurate. This is exactly the bug that
    broke the original app (it silently mismatched 4 vs 8 classes).
    """
    actual = dataset.classes
    if len(actual) != NUM_CLASSES:
        raise ValueError(
            f"Your test folder has {len(actual)} class subfolders "
            f"({actual}), but CLASS_NAMES has {NUM_CLASSES} entries. "
            f"Fix CLASS_NAMES in both evaluate.py and app.py to match."
        )
    print(f"Detected folder order from ImageFolder: {actual}")
    print(f"Configured CLASS_NAMES order:            {CLASS_NAMES}")
    print("Confirm these two lists represent the SAME diseases in the SAME "
          "order (folder abbreviation -> full name), e.g. 'akiec'->'Actinic "
          "Keratosis', 'mel'->'Melanoma', etc. If your folder names sort "
          "differently, edit CLASS_NAMES (in both this file and app.py) to "
          "match the printed folder order exactly.")


def build_stratified_indices(dataset, sample_size, seed=42, min_per_class=5):
    """
    THE FIX for "every model shows the same confusion matrix":
    plain random.shuffle()-then-slice sampling can easily drop rare classes
    entirely out of a small subset (ISIC is heavily imbalanced -- AK, DF and
    VASC are a tiny fraction of the data). Since the exact same subset was
    then reused for all 3 models (fixed seed), all 3 confusion matrices ended
    up with the same missing rows/columns for those classes -- which is what
    made them look identical, even though the models themselves differ.

    This instead samples proportionally from EVERY class folder, with a
    floor of `min_per_class` images per class whenever that many exist, so
    small evaluation runs still actually exercise every disease class.
    """
    random.seed(seed)

    by_class = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)
    for idx_list in by_class.values():
        random.shuffle(idx_list)

    total_available = len(dataset)
    if sample_size >= total_available:
        return list(range(total_available))

    num_classes_present = len(by_class)
    # Start every class off with min_per_class (capped by how many it has),
    # then fill the remainder proportionally to each class's share of data.
    selected = []
    remaining_budget = sample_size
    floor_per_class = {}
    for label, idx_list in by_class.items():
        take = min(min_per_class, len(idx_list), remaining_budget)
        floor_per_class[label] = take

    floor_total = sum(floor_per_class.values())
    if floor_total > sample_size:
        # Not even one min_per_class floor per class fits in the budget --
        # fall back to proportional-only allocation.
        floor_per_class = {label: 0 for label in by_class}
        floor_total = 0

    remaining_budget = sample_size - floor_total
    for label, idx_list in by_class.items():
        selected.extend(idx_list[:floor_per_class[label]])

    # Distribute the rest proportionally to each class's remaining pool size.
    remaining_pools = {
        label: idx_list[floor_per_class[label]:] for label, idx_list in by_class.items()
    }
    total_remaining_pool = sum(len(p) for p in remaining_pools.values())
    if total_remaining_pool > 0 and remaining_budget > 0:
        for label, pool in remaining_pools.items():
            share = int(round(remaining_budget * len(pool) / total_remaining_pool))
            selected.extend(pool[:share])

    selected = list(dict.fromkeys(selected))  # de-dup while preserving order
    random.shuffle(selected)
    return selected[:sample_size]


def build_loader(dataset, batch_size, num_workers, sample_size, seed=42):
    """
    Optionally shrink the dataset to `sample_size` images using stratified
    sampling (see build_stratified_indices) so a full run fits in your time
    budget WITHOUT silently dropping rare classes. Pass sample_size=None to
    use the full test set.
    """
    if sample_size is not None and sample_size < len(dataset):
        indices = build_stratified_indices(dataset, sample_size, seed=seed)
        dataset = Subset(dataset, indices)

        # Show exactly what got sampled, per class, so you can SEE it's not
        # silently skipping rare diseases.
        counts = Counter(dataset.dataset.samples[i][1] for i in indices)
        print(f"Using a stratified subset of {len(indices)} images "
              f"(out of {len(dataset.dataset)} available):")
        for label_idx, cname in enumerate(dataset.dataset.classes):
            print(f"    {cname:30s}: {counts.get(label_idx, 0)} images")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def run_evaluation(model_name, batch_size, num_workers, sample_size, skip_cm):
    weights_file = MODEL_WEIGHTS[model_name]
    if not os.path.exists(weights_file):
        print(f"Weights missing for {model_name}: {weights_file}")
        return None, None

    print(f"\nEvaluating {model_name}... (device={DEVICE})")
    t_start = time.time()

    model = load_model(model_name, NUM_CLASSES, weights_file, DEVICE)
    model.eval()

    full_dataset = ImageFolder(root=TEST_DIR, transform=transform)
    verify_class_order(full_dataset)
    loader = build_loader(full_dataset, batch_size, num_workers, sample_size)

    preds_all, labels_all = [], []
    total_images = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{model_name} inference", unit="batch"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds_all.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            labels_all.extend(labels.numpy())
            total_images += images.size(0)

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average='macro')
    # zero_division=0 so a class with 0 predicted/true samples in this run
    # contributes 0 instead of raising/crashing the whole eval.
    precision_macro = precision_score(labels_all, preds_all, average='macro', zero_division=0)
    recall_macro = recall_score(labels_all, preds_all, average='macro', zero_division=0)

    # Per-class precision/recall computed over the exact same predictions
    # used for accuracy/f1 above (labels=range(NUM_CLASSES) keeps every
    # class in the output even if it had 0 samples in this run).
    precision_per_class = precision_score(
        labels_all, preds_all, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    recall_per_class = recall_score(
        labels_all, preds_all, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    elapsed = time.time() - t_start
    print(f"Results for {model_name.upper()}: accuracy={acc:.4f}  macro_f1={f1:.4f}  "
          f"precision={precision_macro:.4f}  recall={recall_macro:.4f}  "
          f"({total_images} images, {elapsed:.1f}s, "
          f"{elapsed/max(total_images,1):.3f}s/image)")

    print(f"  Per-class precision / recall for {model_name}:")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"    {cname:30s}: precision={precision_per_class[i]:.4f}  "
              f"recall={recall_per_class[i]:.4f}")

    # DIAGNOSTIC: what is this model actually predicting? If every model
    # collapses onto one or two classes, the confusion matrices WILL look
    # near-identical -- but that's a real model/training issue (often class
    # imbalance), not a bug in this script. This makes it visible instead of
    # a mystery.
    pred_counts = Counter(preds_all.tolist())
    print(f"  Prediction distribution for {model_name}:")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"    {cname:30s}: {pred_counts.get(i, 0)}")

    if not skip_cm:
        cm = confusion_matrix(labels_all, preds_all, labels=list(range(NUM_CLASSES)))
        os.makedirs(STATIC_DIR, exist_ok=True)
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        # Saved with the exact filename app.py's /predict endpoint expects
        plt.savefig(os.path.join(STATIC_DIR, f"{model_name}_cm.png"))
        plt.close()

    metrics = {
        "overall_accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1), 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "per_class_precision": {
            CLASS_NAMES[i]: round(float(precision_per_class[i]), 4) for i in range(NUM_CLASSES)
        },
        "per_class_recall": {
            CLASS_NAMES[i]: round(float(recall_per_class[i]), 4) for i in range(NUM_CLASSES)
        },
        "epochs_trained": EPOCHS_TRAINED,
    }
    return metrics, preds_all


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate skin lesion models.")
    p.add_argument("--models", nargs="+", default=list(MODEL_WEIGHTS.keys()),
                    choices=list(MODEL_WEIGHTS.keys()),
                    help="Which model(s) to evaluate. Default: all three.")
    p.add_argument("--sample-size", type=int, default=150,
                    help="Evaluate on a stratified subset of N test images "
                         "instead of the full set (default: 150 -- fast on "
                         "CPU, still covers every class). Pass 0 to use the "
                         "full test set instead.")
    p.add_argument("--batch-size", type=int, default=32,
                    help="DataLoader batch size. Bigger can be faster if RAM allows.")
    p.add_argument("--num-workers", type=int, default=2,
                    help="CPU workers for data loading (parallel image decode/resize).")
    p.add_argument("--skip-cm", action="store_true",
                    help="Skip generating/saving confusion matrix images (saves a "
                         "little time, but this is normally cheap already).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_size = None if args.sample_size == 0 else args.sample_size

    if not os.path.exists(TEST_DIR):
        print(f"Please create a test dataset directory at '{TEST_DIR}' "
              f"structured as {TEST_DIR}/<class_folder>/images...")
    else:
        print(f"Device: {DEVICE}")
        if sample_size:
            print(f"Evaluating on a stratified {sample_size}-image subset "
                  f"(pass --sample-size 0 to use the full test set instead).")
        if DEVICE.type == "cpu" and not sample_size:
            print("Running on CPU with the FULL test set -- this can take a "
                  "while. Consider --sample-size 150 (the default) for a "
                  "quick pass, or --sample-size 0 explicitly for the full run.")

        run_start = time.time()
        all_metrics = {}
        all_preds = {}
        for name in args.models:
            result, preds = run_evaluation(
                name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sample_size=sample_size,
                skip_cm=args.skip_cm,
            )
            if result is not None:
                all_metrics[name] = result
                all_preds[name] = preds

        # DIAGNOSTIC: if two models produced byte-for-byte identical
        # predictions on the same images, that's the actual smoking gun for
        # "every confusion matrix looks the same" -- flag it loudly instead
        # of leaving it to be discovered by eyeballing PNGs.
        model_names = list(all_preds.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a, b = model_names[i], model_names[j]
                if len(all_preds[a]) == len(all_preds[b]) and np.array_equal(all_preds[a], all_preds[b]):
                    print(f"\n⚠️  WARNING: {a} and {b} produced IDENTICAL predictions "
                          f"on every single image. This almost always means "
                          f"they're loading the same weights (check MODEL_WEIGHTS "
                          f"paths above) or the same architecture is being "
                          f"instantiated for both (check models.py). This is why "
                          f"their confusion matrices look the same -- it isn't "
                          f"a plotting bug, the predictions genuinely match.")

        os.makedirs(STATIC_DIR, exist_ok=True)
        metrics_path = os.path.join(STATIC_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        total_elapsed = time.time() - run_start
        print(f"\nSaved real metrics for app.py to read at: {metrics_path}")
        print(f"Total time: {total_elapsed/60:.1f} minutes")