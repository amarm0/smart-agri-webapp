"""
Local-only web interface for the EuroSAT land-classification model trained in
Smart_Agriculture_M3_Optimization.ipynb.

Run locally (NOT deployed to any cloud):
    python app.py
Then open http://127.0.0.1:5000 in your browser.

It expects the two files the notebook saves at the end (Part "Deliverables & summary"):
    model/eurosat_resnet_final.pt
    model/model_config.json
Copy them from the notebook's outputs/models/ folder into this app's model/ folder
(see README.md).
"""
import io
import os

import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image

from model_def import load_trained_model, preprocess_image, CLASS_TO_BOOKLET

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "eurosat_resnet_final.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

_model_state = {"model": None, "classes": None, "img_size": None, "mean": None, "std": None, "error": None}


def get_model():
    """Lazily load the model once, the first time it's needed."""
    if _model_state["model"] is not None or _model_state["error"] is not None:
        return _model_state
    if not (os.path.exists(CONFIG_PATH) and os.path.exists(WEIGHTS_PATH)):
        _model_state["error"] = (
            "Model files not found. Run the notebook to the end first, "
            "then copy eurosat_resnet_final.pt and model_config.json "
            "from outputs/models/ into this app's model/ folder."
        )
        return _model_state
    try:
        model, classes, img_size, mean, std = load_trained_model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)
        _model_state.update(model=model, classes=classes, img_size=img_size, mean=mean, std=std)
    except Exception as exc:  # noqa: BLE001
        _model_state["error"] = f"Failed to load the model: {exc}"
    return _model_state


@app.route("/")
def index():
    state = get_model()
    return render_template("index.html", model_error=state["error"])


@app.route("/predict", methods=["POST"])
def predict():
    state = get_model()
    if state["error"]:
        return jsonify({"ok": False, "error": state["error"]}), 500

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"ok": False, "error": "No image was selected."}), 400

    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read()))
        img.load()  # force-decode now, so a corrupt/unsupported file fails here, not later
    except Exception:
        return jsonify({
            "ok": False,
            "error": "Could not open the file as an image. Try a PNG or JPG file.",
        }), 400

    model = state["model"]
    classes = state["classes"]
    x = preprocess_image(img, state["img_size"], state["mean"], state["std"], device=DEVICE)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    order = probs.argsort()[::-1]
    top = [
        {
            "class_en": classes[i],
            "booklet_en": CLASS_TO_BOOKLET.get(classes[i], "—"),
            "prob": float(probs[i]),
        }
        for i in order[:3]
    ]

    return jsonify({"ok": True, "top": top})


if __name__ == "__main__":
    # debug=True is fine for local dev; set to False for anything closer to "production".
    app.run(host="127.0.0.1", port=5000, debug=True)
