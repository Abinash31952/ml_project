from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)  # ✅ Fix: use __name__

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
    if features[1] > 125:  # Glucose level check
        result = "Diabetic"
    else:
        result = "Not Diabetic"

    return jsonify({'prediction': result})

if __name__ == '__main__':  # ✅ Fix: use __name__ and __main__
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
