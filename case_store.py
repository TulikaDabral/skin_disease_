"""
case_store.py
Tiny JSON-file-backed store that powers the "Connect with a Doctor" feature.

This is intentionally simple (no database, no auth) so it works out of the
box in a local/dev setting:

  - A patient runs a prediction, then clicks "Send to a doctor for review".
    That snapshots the uploaded image + the model's prediction into a new
    "case" with status = "pending".
  - A doctor opens /doctor, sees every pending case (image + AI prediction),
    and submits their own written assessment. The case flips to "reviewed".
  - The patient's page polls /review-status/<case_id> and shows the doctor's
    note as soon as it's in, with no page reload needed.

Swap this out for a real database + auth system before using this with real
patients -- this is a functional demo/prototype layer, not a HIPAA-ready
medical records system.
"""

import os
import json
import time
import uuid
import shutil
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CASES_FILE = os.path.join(DATA_DIR, "cases.json")
CASE_IMAGE_DIR = os.path.join(BASE_DIR, "static", "case_uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CASE_IMAGE_DIR, exist_ok=True)

_lock = threading.Lock()


def _load_all():
    if not os.path.exists(CASES_FILE):
        return {}
    with open(CASES_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_all(cases):
    tmp_path = CASES_FILE + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cases, f, indent=2)
    os.replace(tmp_path, CASES_FILE)


def create_case(image_source_path, prediction_payload, patient_note=""):
    """Snapshot an uploaded image + prediction result as a new pending case.
    Returns the new case dict (includes its id)."""
    case_id = uuid.uuid4().hex[:10]
    ext = os.path.splitext(image_source_path)[1] or ".png"
    dest_filename = f"{case_id}{ext}"
    dest_path = os.path.join(CASE_IMAGE_DIR, dest_filename)
    shutil.copyfile(image_source_path, dest_path)

    case = {
        "id": case_id,
        "created_at": time.time(),
        "image_url": f"/static/case_uploads/{dest_filename}",
        "patient_note": patient_note or "",
        "ai_prediction": prediction_payload.get("prediction"),
        "ai_confidence": prediction_payload.get("confidence"),
        "malignant_flagged": prediction_payload.get("malignant_risk_flagged", False),
        "malignant_class_flagged": prediction_payload.get("malignant_class_flagged"),
        "per_model_predictions": prediction_payload.get("per_model_predictions", {}),
        "status": "pending",  # pending -> reviewed
        "doctor_name": None,
        "doctor_recommendation": None,  # e.g. "urgent_biopsy", "monitor", "benign_likely"
        "doctor_notes": None,
        "reviewed_at": None,
        "messages": [],  # running chat thread: [{sender, text, at}, ...]
    }

    with _lock:
        cases = _load_all()
        cases[case_id] = case
        _save_all(cases)

    return case


def list_cases(status=None):
    with _lock:
        cases = _load_all()
    values = list(cases.values())
    if status:
        values = [c for c in values if c["status"] == status]
    values.sort(key=lambda c: c["created_at"], reverse=True)
    return values


def get_case(case_id):
    with _lock:
        cases = _load_all()
    return cases.get(case_id)


def submit_review(case_id, doctor_name, recommendation, notes):
    with _lock:
        cases = _load_all()
        if case_id not in cases:
            return None
        cases[case_id]["status"] = "reviewed"
        cases[case_id]["doctor_name"] = doctor_name
        cases[case_id]["doctor_recommendation"] = recommendation
        cases[case_id]["doctor_notes"] = notes
        cases[case_id]["reviewed_at"] = time.time()
        _save_all(cases)
        return cases[case_id]


def add_message(case_id, sender, text, doctor_name=None):
    """Append a chat message to a case's thread.
    sender: "patient" or "doctor".
    doctor_name: when sender=="doctor", the name to display (falls back to
    the case's assigned doctor_name, or "Doctor" if neither is set)."""
    with _lock:
        cases = _load_all()
        if case_id not in cases:
            return None
        case = cases[case_id]
        case.setdefault("messages", [])
        display_name = None
        if sender == "doctor":
            display_name = doctor_name or case.get("doctor_name") or "Doctor"
            # Keep the case's doctor_name in sync so the patient side can
            # label the thread even before a formal review is submitted.
            if doctor_name and not case.get("doctor_name"):
                case["doctor_name"] = doctor_name
        case["messages"].append({
            "sender": sender,
            "display_name": display_name,
            "text": text,
            "at": time.time(),
        })
        _save_all(cases)
        return case


def get_messages(case_id):
    case = get_case(case_id)
    if case is None:
        return None
    return case.get("messages", [])
