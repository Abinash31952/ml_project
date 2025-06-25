from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data.get(col) for col in ['Pregnancies', 'Glucose', 'BloodPressure', 
                                          'SkinThickness', 'Insulin', 'BMI', 
                                          'DiabetesPedigreeFunction', 'Age']]
    
    # Scale input
    input_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    # Return readable result
    result = 'Diabetes Detected' if prediction == 1 else 'No Diabetes Detected'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)