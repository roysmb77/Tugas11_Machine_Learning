import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

# ===== Paths =====
MODEL_PATH = "models/ann_churn_model.h5"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
SCHEMA_PATH = "models/feature_schema.json"

# ===== Load artifacts =====
model = tf.keras.models.load_model(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

NUM_COLS = schema["numerical"]
CAT_COLS = schema["categorical"]
CAT_CHOICES = schema["categorical_choices"]

def to_float(x, default=0.0):
    try:
        if x is None:
            return default
        x = str(x).strip()
        if x == "":
            return default
        return float(x)
    except Exception:
        return default

@app.route("/", methods=["GET", "POST"])
def index():
    pred_label = None
    pred_prob = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            row = {}

            # Numerical fields
            for col in NUM_COLS:
                row[col] = to_float(request.form.get(col), default=0.0)
                form_data[col] = request.form.get(col, "")

            # Categorical fields
            for col in CAT_COLS:
                val = request.form.get(col, "")
                if val is None or str(val).strip() == "":
                    val = "Unknown"
                row[col] = val
                form_data[col] = val

            X_raw = pd.DataFrame([row])

            X = preprocessor.transform(X_raw)
            prob = float(model.predict(X, verbose=0).ravel()[0])

            pred_prob = round(prob, 4)
            pred_label = (
                "Churn (Yes) — pelanggan kemungkinan berhenti"
                if prob >= 0.5
                else "No Churn (No) — pelanggan kemungkinan bertahan"
            )

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        num_cols=NUM_COLS,
        cat_cols=CAT_COLS,
        cat_choices=CAT_CHOICES,
        pred_label=pred_label,
        pred_prob=pred_prob,
        error=error,
        form_data=form_data
    )

if __name__ == "__main__":
    # Pastikan folder models/ berisi:
    # - ann_churn_model.h5
    # - preprocessor.pkl
    # - feature_schema.json
    app.run(debug=True)
