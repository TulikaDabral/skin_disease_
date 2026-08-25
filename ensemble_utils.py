"""
ensemble_utils.py
Combines your 3 trained models (ResNet50, EfficientNet-B3, ViT-B/16) into a
single, better, safer prediction than any one model alone.

THREE THINGS THIS ADDS ON TOP OF A SINGLE MODEL:

1. ENSEMBLE (soft voting): average the softmax probabilities of all 3 models.
   Typically +2-5% accuracy/F1 over the best single model, because the three
   architectures make different mistakes on different images.

2. TEMPERATURE CALIBRATION: neural nets are usually overconfident. Dividing
   logits by a learned "temperature" T > 1 before softmax spreads the
   probabilities out so a reported "92%" is actually right ~92% of the time,
   not just an artifact of a peaky softmax. T is fit once on your validation
   set (code included below) and then reused at inference time.

3. RECALL-PRIORITIZED THRESHOLDS: instead of plain argmax, malignant classes
   (Melanoma, BCC, SCC, AK) get a lower bar to be flagged, since missing a
   real melanoma is much worse than a false alarm on a benign lesion. This
   is standard practice in real skin-cancer screening tools - it trades some
   precision for much better recall on the classes that matter most.

USAGE (see the bottom of this file, and the app.py patch instructions):

    from ensemble_utils import EnsemblePredictor
    predictor = EnsemblePredictor(models_dict, CLASS_NAMES, DEVICE)
    result = predictor.predict(img_tensor)
    # result = {
    #   "prediction": "Melanoma",
    #   "confidence": 0.83,
    #   "per_model": {...},
    #   "per_class_probs": {...},
    #   "flagged_malignant": True/False,
    # }
"""

import torch
import torch.nn.functional as F

# Classes where we'd rather over-flag than under-flag (edit to match your
# CLASS_NAMES exactly - these are the ones considered malignant/high-risk).
MALIGNANT_CLASSES = {
    "Melanoma",
    "Basal Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Actinic Keratosis",
}

# Lower = more sensitive (flags more cases as this class even at lower prob).
# Tune these using your validation set - see tune_thresholds() below.
DEFAULT_MALIGNANT_THRESHOLD = 0.30   # flag if malignant-class prob >= 30%
DEFAULT_TEMPERATURE = 1.5            # >1 softens overconfident predictions;
                                      # replace with your fitted value once
                                      # you've run calibrate_temperature()


class EnsemblePredictor:
    def __init__(self, models_dict, class_names, device,
                 temperature=DEFAULT_TEMPERATURE,
                 malignant_threshold=DEFAULT_MALIGNANT_THRESHOLD,
                 weights=None):
        """
        models_dict: {"resnet50": model, "efficientnet-b3": model, "visiontransformer": model}
                     all already .eval() and on `device`
        weights: optional dict of per-model ensemble weights, e.g.
                 {"resnet50": 1.0, "efficientnet-b3": 1.2, "visiontransformer": 1.0}
                 (give more weight to whichever model has the best val macro-F1)
                 defaults to equal weighting.
        """
        self.models = models_dict
        self.class_names = class_names
        self.device = device
        self.temperature = temperature
        self.malignant_threshold = malignant_threshold
        self.weights = weights or {name: 1.0 for name in models_dict}
        w_sum = sum(self.weights.values())
        self.weights = {k: v / w_sum for k, v in self.weights.items()}  # normalize

    @torch.no_grad()
    def _calibrated_probs(self, model, img_tensor):
        logits = model(img_tensor)
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits[0], dim=0)

    @torch.no_grad()
    def predict(self, img_tensor):
        img_tensor = img_tensor.to(self.device)

        per_model_probs = {}
        ensemble_probs = torch.zeros(len(self.class_names), device=self.device)

        for name, model in self.models.items():
            probs = self._calibrated_probs(model, img_tensor)
            per_model_probs[name] = {
                self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
            }
            ensemble_probs += self.weights[name] * probs

        pred_idx = int(torch.argmax(ensemble_probs).item())
        pred_class = self.class_names[pred_idx]
        confidence = float(ensemble_probs[pred_idx])

        # Recall-prioritized override: if any malignant class crosses the
        # lower threshold, surface it even if it isn't the argmax winner.
        # We pick the highest-probability malignant class that clears the bar.
        flagged_malignant = False
        malignant_override = None
        best_malignant_prob = -1.0
        for i, cname in enumerate(self.class_names):
            if cname in MALIGNANT_CLASSES:
                p = float(ensemble_probs[i])
                if p >= self.malignant_threshold and p > best_malignant_prob:
                    best_malignant_prob = p
                    malignant_override = cname
                    flagged_malignant = True

        final_prediction = pred_class
        final_confidence = confidence
        if flagged_malignant and malignant_override != pred_class:
            # Don't silently overwrite - report both, let the UI show the
            # primary prediction AND the malignant-risk flag transparently.
            final_prediction = pred_class  # keep argmax as "prediction"
        # (see app.py patch below - both fields get returned to the frontend)

        return {
            "prediction": final_prediction,
            "confidence": final_confidence,
            "per_class_probs": {
                self.class_names[i]: float(ensemble_probs[i]) for i in range(len(self.class_names))
            },
            "per_model": per_model_probs,
            "flagged_malignant": flagged_malignant,
            "malignant_class_flagged": malignant_override,
            "malignant_class_probability": best_malignant_prob if flagged_malignant else None,
        }


# --------------------------------------------------------------------------
# ONE-TIME CALIBRATION SCRIPTS - run these on Kaggle/locally against your
# labeled validation set, then hardcode the resulting numbers as
# DEFAULT_TEMPERATURE / DEFAULT_MALIGNANT_THRESHOLD above.
# --------------------------------------------------------------------------
def calibrate_temperature(model, val_loader, device, max_iter=50):
    """
    Fits a single scalar temperature T minimizing NLL on the validation set.
    Run once per model (or once on the ensemble's combined logits) after
    training finishes. Requires a labeled val_loader yielding (images, labels).
    """
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits_list.append(model(images).cpu())
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    nll_criterion = torch.nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    fitted_T = temperature.item()
    print(f"Fitted temperature: {fitted_T:.3f}")
    return fitted_T


def tune_thresholds(val_probs, val_labels, class_names, target_classes,
                     recall_target=0.90):
    """
    For each class in target_classes, finds the lowest probability threshold
    that achieves at least `recall_target` recall on your validation set.
    val_probs: (N, num_classes) array of predicted probabilities
    val_labels: (N,) array of true integer labels
    """
    import numpy as np
    thresholds = {}
    for cname in target_classes:
        c_idx = class_names.index(cname)
        class_mask = (val_labels == c_idx)
        if class_mask.sum() == 0:
            continue
        class_probs = val_probs[class_mask, c_idx]
        sorted_probs = np.sort(class_probs)
        n_needed = int(np.ceil(recall_target * len(sorted_probs)))
        # threshold at the (len - n_needed)-th smallest prob keeps that many recalled
        idx = max(0, len(sorted_probs) - n_needed)
        thresholds[cname] = float(sorted_probs[idx])
        print(f"{cname}: threshold={thresholds[cname]:.3f} "
              f"gives ~{recall_target*100:.0f}% recall on {class_mask.sum()} val samples")
    return thresholds