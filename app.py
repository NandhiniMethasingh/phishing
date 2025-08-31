from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

# ================== LOAD MODEL & FEATURES ==================
try:
    model = joblib.load("best_model.pkl")
    FEATURE_COLUMNS = joblib.load("feature_columns.pkl")
    print("✅ Model and feature columns loaded successfully")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model or feature columns: {e}")

# ================== FLASK APP ==================
app = Flask(__name__)
CORS(app)

# --- Home route (serves frontend) ---
@app.route("/", methods=["GET"])
def home():
    try:
        return render_template("index.html")  # requires templates/index.html
    except Exception:
        return "<h3>✅ Flask server is running. Place your index.html inside /templates folder.</h3>"

# --- Health check ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# --- Prediction endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid or missing JSON input"}), 400

        # Build feature dictionary (fill missing with 0)
        features = {col: data.get(col, 0) for col in FEATURE_COLUMNS}

        # Convert to DataFrame (1 row)
        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        # Predict
        prediction = model.predict(X)[0]
        label = "phishing" if int(prediction) == 1 else "legitimate"

        return jsonify({
            "prediction": int(prediction),
            "label": label,
            "used_features": features
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ================== MAIN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
