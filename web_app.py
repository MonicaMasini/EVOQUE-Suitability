from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import os

# Load the Excel file
df = pd.read_excel("Annular_Sizing_Model_Predictions_Updated (1).xlsx", engine="openpyxl")

# Prepare features and target
X = df[["S-L dimension (mm)", "A-P Dimension (mm)"]]
y = df["Target"]

# Cross-validated Random Forest
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, n_jobs=-1)
grid_search.fit(X, y)
model = grid_search.best_estimator_

# Flask app
app = Flask(__name__)

# HTML form template
form_html = """
<!DOCTYPE html>
<html>
<head>
    <title>EVOQUE Suitability Prediction</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        label { font-weight: bold; }
        .result { color: red; font-weight: bold; font-size: 1.2em; }
        .probability { font-style: italic; }
    </style>
</head>
<body>
    <h1>EVOQUE Suitability Prediction</h1>
    <form method="POST">
        <label>S-L Diameter (mm):</label>
        <input type="number" step="any" name="sl" required><br><br>
        <label>A-P Diameter (mm):</label>
        <input type="number" step="any" name="ap" required><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <div class="result">Prediction: {{ prediction }}</div>
        <div class="probability">Probability: {{ probability }}% <span>(Probability of Screen Fail)</span></div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        sl = float(request.form["sl"])
        ap = float(request.form["ap"])
        input_df = pd.DataFrame([[sl, ap]], columns=["S-L dimension (mm)", "A-P Dimension (mm)"])
        prob = model.predict_proba(input_df)[0][1]

        if prob >= 0.45:
            prediction = "Screen Fail"
        elif prob > 0.4:
            prediction = "CT Recommended"
        else:
            prediction = "Suitable"

        probability = round(prob * 100, 1)

    return render_template_string(form_html, prediction=prediction, probability=probability)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sl = data.get("S-L dimension (mm)")
    ap = data.get("A-P Dimension (mm)")

    if sl is None or ap is None:
        return jsonify({"error": "Missing input values"}), 400

    input_df = pd.DataFrame([[sl, ap]], columns=["S-L dimension (mm)", "A-P Dimension (mm)"])
    prob = model.predict_proba(input_df)[0][1]

    if prob >= 0.45:
        prediction = "Screen Fail"
    elif prob > 0.4:
        prediction = "CT Recommended"
    else:
        prediction = "Suitable"

    return jsonify({
        "prediction": prediction,
        "probability": f"{round(prob * 100, 1)}%"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
