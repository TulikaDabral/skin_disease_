from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import traceback
import cv2
import numpy as np

from models import load_model
from explainers import GradCAM, get_lime_explanation, get_shap_explanation
from ensemble_utils import EnsemblePredictor
import case_store

app = Flask(__name__)

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
METRICS_FILE = os.path.join(STATIC_DIR, 'metrics.json')

# --- CONFIGURATION ---
# IMPORTANT: this order MUST exactly match the alphabetical class order your
# training script's ImageFolder(...).classes produced (that's how PyTorch
# assigns label indices 0..N-1). This is the standard ISIC 2019 8-class set.
# Run evaluate.py first -- it will hard-fail if your actual folder order
# doesn't match this list, so you can catch a mismatch before it reaches the UI.
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

models_dict = {}
ensemble_predictor = None  # built lazily once all 3 models are loaded


def get_ensemble_predictor():
    """Loads all 3 models (if not already loaded) and builds one shared
    EnsemblePredictor instance, reused across requests."""
    global ensemble_predictor
    if ensemble_predictor is None:
        for name in ["resnet50", "efficientnet-b3", "visiontransformer"]:
            get_or_load_model(name)  # populates models_dict[name]
        ensemble_predictor = EnsemblePredictor(
            models_dict=models_dict,
            class_names=CLASS_NAMES,
            device=DEVICE,
            # TODO: replace with your fitted temperature from
            # ensemble_utils.calibrate_temperature() once you've run it
            # against your validation set.
            temperature=1.5,
            # TODO: give more weight to whichever single model has the best
            # validation macro-F1 from your evaluate.py run, e.g.
            # weights={"resnet50": 1.0, "efficientnet-b3": 1.2, "visiontransformer": 1.0}
        )
    return ensemble_predictor


def load_metrics():
    """Load REAL metrics computed by evaluate.py. No hardcoded numbers.

    Called fresh on every /predict request (see below) rather than once at
    startup, so re-running evaluate.py while the Flask server is still
    running updates the UI immediately -- no server restart needed.
    """
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    print("WARNING: static/metrics.json not found. Run evaluate.py for each "
          "model first, otherwise the UI will show 'N/A' for accuracy/F1.")
    return {}


def get_or_load_model(model_name):
    if model_name not in models_dict:
        if model_name == "efficientnet-b3":
            weights_file = os.path.join(BASE_DIR, "efficientnet-b3_best.pth")
        elif model_name == "visiontransformer":
            weights_file = os.path.join(BASE_DIR, "visiontransformer_best.pth")
        else:
            weights_file = os.path.join(BASE_DIR, "resnet50_best.pth")

        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Missing weight file asset: {weights_file}")

        print(f"Loading {model_name} weights from {weights_file}...")
        checkpoint = torch.load(weights_file, map_location=DEVICE)

        saved_classes = NUM_CLASSES
        if model_name == "resnet50" and "fc.bias" in checkpoint:
            saved_classes = checkpoint["fc.bias"].shape[0]
        elif model_name == "efficientnet-b3" and "classifier.1.bias" in checkpoint:
            saved_classes = checkpoint["classifier.1.bias"].shape[0]
        elif model_name == "visiontransformer" and "heads.head.bias" in checkpoint:
            saved_classes = checkpoint["heads.head.bias"].shape[0]

        # NOTE: we deliberately do NOT truncate/slice the final layer anymore.
        # A previous version of this file sliced the trained classifier down
        # to NUM_CLASSES rows whenever there was a mismatch -- that silently
        # destroyed the model's real weights and is why predictions always
        # collapsed to one class. If the checkpoint doesn't match, that's a
        # real configuration bug and we want to know about it immediately.
        if saved_classes != NUM_CLASSES:
            raise ValueError(
                f"Checkpoint for '{model_name}' was trained with "
                f"{saved_classes} output classes, but CLASS_NAMES in app.py "
                f"has {NUM_CLASSES} entries. These must match exactly -- "
                f"check the class order you trained with (see evaluate.py) "
                f"and update CLASS_NAMES accordingly."
            )

        model = load_model(model_name, saved_classes, weights_file, DEVICE)
        model.eval()
        models_dict[model_name] = model
    return models_dict[model_name]


def normalize_heatmap_for_cv2(heatmap):
    arr = np.asarray(heatmap)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        min_v, max_v = float(arr.min()), float(arr.max())
        arr = np.zeros_like(arr, dtype=np.uint8) if max_v - min_v < 1e-8 else \
            ((arr - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    return arr


def overlay_heatmap_on_image(heatmap, original_pil_img, alpha=0.45):
    """Blend a (H,W) or colorized heatmap on top of the actual uploaded
    lesion photo (resized to 224x224 to match the model's input) so the
    explanation is visible *in context* instead of a floating colour blob.
    Returns a BGR uint8 image ready for cv2.imwrite.
    """
    colored = normalize_heatmap_for_cv2(heatmap)  # BGR, 224x224x3 (uint8)

    base_rgb = np.array(original_pil_img.resize((224, 224)))
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(colored, alpha, base_bgr, 1 - alpha, 0)
    return overlay


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/metrics", methods=["GET"])
def metrics():
    """Lets the front end fetch accuracy/F1/epochs on page load (before any
    prediction is run) and refresh them any time, e.g. after re-running
    evaluate.py -- no server restart needed."""
    return jsonify(load_metrics())


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "model" not in request.form:
        return jsonify({"error": "Missing input image file uploads"}), 400

    file = request.files["image"]
    raw_model_name = request.form["model"].lower().strip()

    if "resnet" in raw_model_name:
        selected_model_name = "resnet50"
    elif "efficient" in raw_model_name:
        selected_model_name = "efficientnet-b3"
    else:
        selected_model_name = "visiontransformer"

    try:
        img = Image.open(file.stream).convert("RGB")
        # Save the original uploaded image to static/uploaded_image.png
        uploaded_img_path = os.path.join(STATIC_DIR, "uploaded_image.png")
        img.save(uploaded_img_path)

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Single-model path still used for Grad-CAM/LIME/SHAP (those need one
        # concrete model+layer to explain), but the actual prediction shown
        # to the user comes from the calibrated 3-model ensemble below.
        model = get_or_load_model(selected_model_name)

        predictor = get_ensemble_predictor()
        ensemble_result = predictor.predict(img_tensor)

        pred_class = ensemble_result["prediction"]
        pred_idx = CLASS_NAMES.index(pred_class)
        confidence = torch.tensor(ensemble_result["confidence"])

        # Re-read metrics.json fresh each request (cheap: it's a tiny JSON
        # file) so a re-run of evaluate.py shows up without restarting Flask.
        current_metrics = load_metrics()
        stats = current_metrics.get(
            selected_model_name,
            {
                "overall_accuracy": None,
                "macro_f1": None,
                "macro_precision": None,
                "macro_recall": None,
                "epochs_trained": None,
            }
        )

        cm_url = f"/static/{selected_model_name}_cm.png"

        # 1. Grad-CAM (Fast, compute instantly)
        gradcam_url = None
        cam = None
        try:
            target_layer = None
            if selected_model_name == "resnet50":
                target_layer = model.layer4[-1]
            elif selected_model_name == "efficientnet-b3":
                target_layer = model.features[-1]
            elif selected_model_name == "visiontransformer":
                # NOTE: must be the SECOND-to-last encoder block, not the last.
                # The classifier head only reads the CLS token of the final
                # block's output, so the gradient of the loss w.r.t. the
                # *patch-token* outputs of that final block is exactly zero
                # (they never reach anything downstream). Hooking layers[-1]
                # silently produced an all-zero CAM. layers[-2]'s output
                # feeds every token of the last block via self-attention, so
                # patch tokens there DO carry a real gradient.
                target_layer = model.encoder.layers[-2]

            if target_layer is not None:
                cam = GradCAM(model, target_layer)
                heatmap = cam.generate(img_tensor, class_idx=pred_idx)
                heatmap_img = overlay_heatmap_on_image(heatmap, img)
                gradcam_path = os.path.join(STATIC_DIR, f"{selected_model_name}_gradcam.png")
                if cv2.imwrite(gradcam_path, heatmap_img):
                    gradcam_url = f"/static/{selected_model_name}_gradcam.png"
        except Exception as e:
            print(f"Grad-CAM processing skipped or caught error: {e}")
            traceback.print_exc()
        finally:
            if cam is not None:
                cam.remove_hooks()

        # 2. LIME and SHAP are deferred to the /explain endpoint for on-demand calculation.
        lime_url = None
        shap_url = None

        return jsonify({
            "model_used": f"ensemble (explanations shown from {selected_model_name})",
            "prediction": pred_class,
            "confidence": f"{confidence.item() * 100:.2f}%",
            "malignant_risk_flagged": ensemble_result["flagged_malignant"],
            "malignant_class_flagged": ensemble_result["malignant_class_flagged"],
            "per_model_predictions": {
                name: max(probs, key=probs.get)
                for name, probs in ensemble_result["per_model"].items()
            },
            "per_class_probabilities": ensemble_result["per_class_probs"],
            "model_metrics": {
                "accuracy": stats["overall_accuracy"],
                "f1_score": stats["macro_f1"],
                "precision": stats.get("macro_precision"),
                "recall": stats.get("macro_recall"),
                "epochs": stats["epochs_trained"]
            },
            "confusion_matrix_url": cm_url,
            "xai_explanations": {
                "gradcam_url": gradcam_url,
                "lime_url": lime_url,
                "shap_url": shap_url
            }
        })

    except Exception as err:
        traceback.print_exc()
        return jsonify({"error": str(err)}), 500


@app.route("/request-review", methods=["POST"])
def request_review():
    """Patient side: snapshot the last uploaded image + AI prediction into a
    new pending case that a doctor can pick up on the /doctor dashboard."""
    uploaded_img_path = os.path.join(STATIC_DIR, "uploaded_image.png")
    if not os.path.exists(uploaded_img_path):
        return jsonify({"error": "No uploaded image found. Run a prediction first."}), 400

    try:
        per_model_raw = request.form.get("per_model_predictions", "{}")
        try:
            per_model = json.loads(per_model_raw)
        except json.JSONDecodeError:
            per_model = {}

        prediction_payload = {
            "prediction": request.form.get("prediction"),
            "confidence": request.form.get("confidence"),
            "malignant_risk_flagged": request.form.get("malignant_risk_flagged") == "true",
            "malignant_class_flagged": request.form.get("malignant_class_flagged") or None,
            "per_model_predictions": per_model,
        }
        patient_note = request.form.get("patient_note", "")

        case = case_store.create_case(uploaded_img_path, prediction_payload, patient_note)
        return jsonify({"case_id": case["id"], "status": case["status"]})
    except Exception as err:
        traceback.print_exc()
        return jsonify({"error": str(err)}), 500


@app.route("/review-status/<case_id>", methods=["GET"])
def review_status(case_id):
    """Patient side: poll this to find out when a doctor has reviewed the case."""
    case = case_store.get_case(case_id)
    if case is None:
        return jsonify({"error": "Unknown case id"}), 404
    return jsonify(case)


@app.route("/case/<case_id>/messages", methods=["GET"])
def get_case_messages(case_id):
    """Patient side: poll this for new chat messages (also included inline
    in /review-status, this endpoint exists for lighter-weight polling)."""
    messages = case_store.get_messages(case_id)
    if messages is None:
        return jsonify({"error": "Unknown case id"}), 404
    return jsonify(messages)


@app.route("/case/<case_id>/messages", methods=["POST"])
def post_patient_message(case_id):
    """Patient side: send a chat message to the doctor on this case."""
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Message can't be empty"}), 400
    case = case_store.add_message(case_id, "patient", text)
    if case is None:
        return jsonify({"error": "Unknown case id"}), 404
    return jsonify(case)


@app.route("/doctor/api/case/<case_id>/messages", methods=["POST"])
def post_doctor_message(case_id):
    """Doctor side: reply in the chat thread for this case."""
    text = request.form.get("text", "").strip()
    doctor_name = request.form.get("doctor_name", "").strip() or None
    if not text:
        return jsonify({"error": "Message can't be empty"}), 400
    case = case_store.add_message(case_id, "doctor", text, doctor_name=doctor_name)
    if case is None:
        return jsonify({"error": "Unknown case id"}), 404
    return jsonify(case)


@app.route("/doctor", methods=["GET"])
def doctor_dashboard():
    """Doctor-facing dashboard: lists every case waiting for a human review."""
    return render_template("doctor.html")


@app.route("/doctor/api/cases", methods=["GET"])
def doctor_list_cases():
    status = request.args.get("status")  # "pending", "reviewed", or omitted for all
    return jsonify(case_store.list_cases(status))


@app.route("/doctor/api/case/<case_id>", methods=["GET"])
def doctor_get_case(case_id):
    """Lightweight single-case fetch, used by the doctor dashboard's chat
    panel to poll without re-fetching the whole queue every few seconds."""
    case = case_store.get_case(case_id)
    if case is None:
        return jsonify({"error": "Unknown case id"}), 404
    return jsonify(case)


@app.route("/doctor/api/review/<case_id>", methods=["POST"])
def doctor_submit_review(case_id):
    doctor_name = request.form.get("doctor_name", "").strip() or "Unnamed reviewer"
    recommendation = request.form.get("recommendation", "").strip()
    notes = request.form.get("notes", "").strip()

    if not recommendation:
        return jsonify({"error": "A recommendation is required"}), 400

    case = case_store.submit_review(case_id, doctor_name, recommendation, notes)
    if case is None:
        return jsonify({"error": "Unknown case id"}), 404
    return jsonify(case)


@app.route("/explain", methods=["POST"])
def explain():
    if "type" not in request.form or "model" not in request.form:
        return jsonify({"error": "Missing type or model parameter"}), 400

    explain_type = request.form["type"].lower().strip()
    raw_model_name = request.form["model"].lower().strip()

    if "resnet" in raw_model_name:
        selected_model_name = "resnet50"
    elif "efficient" in raw_model_name:
        selected_model_name = "efficientnet-b3"
    else:
        selected_model_name = "visiontransformer"

    uploaded_img_path = os.path.join(STATIC_DIR, "uploaded_image.png")
    if not os.path.exists(uploaded_img_path):
        return jsonify({"error": "No uploaded image found to explain. Run prediction first."}), 400

    try:
        img = Image.open(uploaded_img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        model = get_or_load_model(selected_model_name)

        # Run model to get prediction class for explanation target
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = int(torch.argmax(output, dim=1).item())

        explain_url = None

        if explain_type == "lime":
            lime_res = get_lime_explanation(model, img_tensor, transform, DEVICE)
            lime_res = normalize_heatmap_for_cv2(lime_res)
            if lime_res.ndim == 3 and lime_res.shape[-1] == 3:
                lime_res = cv2.cvtColor(lime_res, cv2.COLOR_RGB2BGR)
            lime_path = os.path.join(STATIC_DIR, f"{selected_model_name}_lime.png")
            if cv2.imwrite(lime_path, lime_res):
                explain_url = f"/static/{selected_model_name}_lime.png"

        elif explain_type == "shap":
            shap_heat = get_shap_explanation(model, img_tensor, pred_idx, DEVICE)
            shap_img = overlay_heatmap_on_image(shap_heat, img)
            shap_path = os.path.join(STATIC_DIR, f"{selected_model_name}_shap.png")
            if cv2.imwrite(shap_path, shap_img):
                explain_url = f"/static/{selected_model_name}_shap.png"
        else:
            return jsonify({"error": f"Unknown explanation type: {explain_type}"}), 400

        if explain_url is None:
            return jsonify({"error": f"Failed to generate {explain_type} explanation"}), 500

        return jsonify({"url": explain_url})

    except Exception as err:
        traceback.print_exc()
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    # threaded=True matters here: SHAP explanations can take 10-35s+, and
    # without threading a single slow /explain request would block every
    # other request (chat polling, review-status polling, a second user's
    # /predict) until it finished -- which looks like the UI "hanging" even
    # though the server is fine. Flask's dev server is single-request-at-a-
    # time by default, so this flag is what actually fixes that.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)