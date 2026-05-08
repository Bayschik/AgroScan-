"""
Flask API-сервер системы диагностики растений
"""

import os
import sys
import json
import traceback
import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ─── пути ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from knowledge_base import DISEASES, SYMPTOM_LIST
from diagnosis_engine import DiagnosisEngine
from image_analyzer import PlantImageAnalyzer

# ─── Flask ────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

engine = DiagnosisEngine()
analyzer = PlantImageAnalyzer()

ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ─────────────────────────────────────────────────────
# UI СТРАНИЦЫ
# ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect")
def detect_page():
    return render_template("detect.html")


# ─────────────────────────────────────────────────────
# API HEALTH
# ─────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────────────
# SYMPTOMS / DISEASES
# ─────────────────────────────────────────────────────

@app.route("/api/symptoms")
def get_symptoms():
    return jsonify({
        "symptoms": [
            {"id": k, "label": v["label"], "icon": v["icon"]}
            for k, v in SYMPTOM_LIST.items()
        ]
    })


@app.route("/api/diseases")
def get_diseases():
    return jsonify({
        "diseases": [
            {
                "id": k,
                "name": v["name"],
                "name_en": v["name_en"],
                "severity": v["severity"],
                "description": v["description"],
            }
            for k, v in DISEASES.items()
            if k != "healthy"
        ]
    })


# ─────────────────────────────────────────────────────
# DIAGNOSE API (JSON)
# ─────────────────────────────────────────────────────

@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    try:
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", [])

        if not isinstance(symptoms, list):
            return jsonify({"error": "symptoms должен быть списком"}), 400

        valid = [s for s in symptoms if s in SYMPTOM_LIST]

        if len(valid) != len(symptoms):
            return jsonify({"error": "Есть неизвестные симптомы"}), 400

        result = engine.diagnose(symptoms=valid)
        result["plant_name"] = data.get("plant_name", "Неизвестное растение")

        return jsonify(result)

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Server error"}), 500


# ─────────────────────────────────────────────────────
# WEB FORM DIAGNOSE
# ─────────────────────────────────────────────────────

@app.route("/web-diagnose", methods=["POST"])
def web_diagnose():
    text = request.form.get("symptoms", "")
    result = engine.diagnose(symptoms=[text])
    return f"Результат: {result}"


# ─────────────────────────────────────────────────────
# UPLOAD IMAGE
# ─────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["image"]

    if not allowed_file(file.filename):
        return jsonify({"error": "bad format"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return jsonify({"status": "image received"})


# ─────────────────────────────────────────────────────
# IMAGE + DIAGNOSIS
# ─────────────────────────────────────────────────────

@app.route("/api/analyze-image", methods=["POST"])
def analyze_image():
    try:
        data = request.get_json(force=True)

        img_source = data.get("image_base64")
        symptoms = data.get("symptoms", [])
        plant_name = data.get("plant_name", "Неизвестное растение")

        if not img_source:
            return jsonify({"error": "no image"}), 400

        cv_result = analyzer.analyze(img_source)

        image_symptoms = cv_result.get("detected_symptoms", [])
        health_score = cv_result.get("health_score")

        valid_syms = [s for s in symptoms if s in SYMPTOM_LIST]

        diag = engine.diagnose(
            symptoms=valid_syms,
            image_symptoms=image_symptoms,
            image_score=health_score
        )

        diag["plant_name"] = plant_name

        return jsonify({
            **diag,
            "opencv_analysis": cv_result
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "server error"}), 500


# ─────────────────────────────────────────────────────
# START SERVER (ОДИН РАЗ!)
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🌱 Plant Diagnostics запускается...")
    app.run(debug=True, use_reloader=False)