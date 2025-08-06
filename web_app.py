from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Load the Excel file
df = pd.read_excel("Annular_Sizing_Model_Predictions_Updated (1).xlsx", engine="openpyxl")

# Select features and target
X = df[["S-L dimension (mm)", "A-P Dimension (mm)"]]
y = df["Target"]

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create Flask app
app = Flask(__name__)

# HTML template for the form
form_html = """
<!DOCTYPE html>
<html>
<head>
    <title>EVOQUE Suitability Prediction</title>
</head>
<body>
    <h1>EVOQUE Suitability Prediction</h1>
    <form method="POST">
        <label for="sl">S-L dimension (mm):</label>
        <input type="number" step="any" name="sl" required><br><br>
        <label for="ap">A-P Dimension (mm):</label>
        <input type="number" step="any" name="ap" required><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h2>Prediction: {{ prediction }}</h2>
        <p>Probability: {{ probability }}</p>
    {% endif %}
</body>
</html>
"""

# Homepage with form
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

        probability = round(prob, 3)

    return render_template_string(form_html, prediction=prediction, probability=probability)

# API endpoint
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
        "probability": round(prob, 3)
    })

# Bind to 0.0.0.0 and use PORT from environment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
