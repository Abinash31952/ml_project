from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(_name_)

# Load model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return "✅ ML API is live and ready!", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return jsonify({'prediction': result})

if _name_ == '_main_':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)