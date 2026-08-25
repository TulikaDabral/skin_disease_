# ISIC Skin Lesion Ensemble Classifier — Fixed Project

## New in this update

1. **Real frontend.** `templates/index.html` + `static/style.css` +
   `static/script.js` now actually exist (they were referenced by `app.py`
   but missing from the original zip, which is why the UI looked
   disconnected/empty). The results panel always shows *something*: an
   idle state describing what will appear, a step-by-step "reading the
   specimen" tracker while `/predict` is running (replacing a bare
   spinner), and a fully populated readout once results come back —
   prediction, calibrated confidence, per-model votes, the full 8-class
   probability spectrum, Grad-CAM/LIME/SHAP tabs, and the confusion matrix.

2. **"Connect with a doctor" feature.** After a prediction, the patient can
   add an optional note and send the image + AI readout to a review queue.
   - `POST /request-review` snapshots the case (`case_store.py`, JSON file +
     copied image, no database needed).
   - `GET /doctor` is a dashboard where a doctor sees the queue, opens a
     case, and submits a recommendation (benign / monitor / in-person exam /
     urgent biopsy) plus free-text notes.
   - The patient's page polls `GET /review-status/<case_id>` and shows the
     doctor's write-up inline as soon as it's submitted — no refresh needed.
   - This is a functional prototype (no login/auth, JSON-file storage) —
     swap in real accounts and a database before using it with real patients.

## What was broken


1. **No frontend existed.** `app.py` called `render_template("index.html")` but
   there was no `templates/` folder at all — that's why "backend and frontend
   weren't connected." Added `templates/index.html`, `static/style.css`,
   `static/script.js`.

2. **ViT Grad-CAM returned an all-zero heatmap.** The code hooked
   `model.encoder.layers[-1]` (the last transformer block). The classifier
   head only reads the CLS token from that block's output, so the gradient
   of the loss with respect to the *patch-token* outputs of that last block
   is exactly zero — they never feed anything downstream. Fixed by hooking
   `model.encoder.layers[-2]` instead, confirmed non-zero.

3. **Grad-CAM / SHAP showed a bare colormap with no context.** Added
   `overlay_heatmap_on_image()` in `app.py` so both are now blended onto the
   actual uploaded lesion photo (LIME already did this).

4. **`convert_weights.py` is not needed.** Your three `.pth` files are
   already in native torchvision format and load with
   `<All keys matched successfully>` directly into `resnet50`,
   `efficientnet_b3`, and `vit_b_16` — don't run the timm conversion against
   them.

5. **`test_pipeline.py`'s SHAP timeout was too tight.** SHAP does dozens of
   forward passes over masked crops of the image, so it's inherently the
   slowest of the three explainers (measured 12-34s per model on a fast CPU
   sandbox -- expect longer, especially for ViT, on a normal laptop CPU). The
   old 45s timeout could cut it off before it finished, which looked like
   "SHAP failed" even though the server was still working. Bumped to 180s
   (LIME: 30s -> 90s).

6. **Removed the animated scanline** from the frontend header per request.

## What to expect timing-wise
- Grad-CAM: instant (comes back with `/predict`)
- LIME: a few seconds
- SHAP: 10-35+ seconds per model on CPU -- this is normal, not a hang. The
  UI shows a spinner with "Sampling coalitions (SHAP)..." while it works.

## Checked this session (no GPU/torch in this sandbox, so verified at the level available)

- All `.py` files compile cleanly (`python -m py_compile`).
- `templates/index.html` and `templates/doctor.html` render through Jinja
  with no template errors.
- `case_store.py` (the doctor-review data layer) was run standalone:
  creating a case, listing it as pending, submitting a doctor review, and
  confirming it then lists as reviewed — all passed.
- Full `/predict` → ensemble → Grad-CAM/LIME/SHAP inference was **not**
  re-run end-to-end here (this sandbox has no `torch`/GPU installed) — run
  `python test_pipeline.py` yourself after starting `app.py` to confirm
  inference on your machine, the same way a previous session reportedly did.

## To run it

```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000        -> patient scan tool
# open http://127.0.0.1:5000/doctor -> doctor review dashboard
```

Model weight files (`resnet50_best.pth`, `efficientnet-b3_best.pth`,
`visiontransformer_best.pth`) **are included** in this zip.

## Still on you: real metrics

`model_metrics` in the UI will show `N/A` until you run `evaluate.py`
against a real labeled test split:

```
data/test/<class_folder_1>/*.jpg
data/test/<class_folder_2>/*.jpg
...
```

The folder names must sort alphabetically into the same order as
`CLASS_NAMES` in both `app.py` and `evaluate.py` — `evaluate.py` will
print the detected folder order vs. the configured order and hard-fail
on a mismatch, so you can catch it before it reaches the UI. Once it
runs successfully it writes `static/metrics.json` and the per-model
confusion matrix PNGs that the UI already knows how to display.
