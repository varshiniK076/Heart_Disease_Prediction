from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# ================= LOAD MODEL =================
with open("Heart_Disease_Prediction.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler if exists
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Input features in same order as training
        input_features = [
            float(request.form.get("age")),
            float(request.form.get("sex")),
            float(request.form.get("cp")),
            float(request.form.get("trestbps")),
            float(request.form.get("chol")),
            float(request.form.get("fbs")),
            float(request.form.get("restecg")),
            float(request.form.get("thalach")),
            float(request.form.get("exang")),
            float(request.form.get("oldpeak")),
            float(request.form.get("slope")),
            float(request.form.get("ca")),
            float(request.form.get("thal"))
        ]

        input_array = np.array(input_features).reshape(1, -1)

        # ================= SCALE NUMERICAL COLUMNS =================
        # Only continuous numerical features
        num_idx = [0, 3, 4, 7, 9]  # age, trestbps, chol, thalach, oldpeak
        if scaler is not None:
            input_array[:, num_idx] = scaler.transform(input_array[:, num_idx])

        # ================= PREDICTION =================
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_array)[0][1] * 100
            prob = round(prob, 2)
        else:
            # fallback if predict_proba not available
            pred = model.predict(input_array)[0]
            prob = 100 if pred == 1 else 0

        return jsonify({"probability": prob})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
