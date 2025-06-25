from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# ✅ Load your actual model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/', methods=['GET'])
def home():
    return "✅ ML API is live and ready!", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # 📦 Convert the input to a NumPy array
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

    # 🧼 Scale input and predict using model
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    # 🩺 Return prediction result
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)