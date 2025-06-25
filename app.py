from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "✅ ML API is live and ready!", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ])

    # ⚠ TEMPORARY: Dummy prediction logic
    # This is a placeholder until model.pkl and scaler.pkl are available
    if features[1] > 125:  # e.g., Glucose > 125
        result = "Diabetic"
    else:
        result = "Not Diabetic"

    return jsonify({'prediction': result})

if _name_ == '_main_':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)