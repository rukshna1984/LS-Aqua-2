from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Allows your Lovable frontend to talk to this backend

# Load your trained model file
model = joblib.load('filter_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
 
    # Get values sent from frontend (defaulting if missing)
    current = float(data.get('current_A', 0.45))
    flow = float(data.get('flow_Lmin', 0.7))
 
    # Predict health score using the model
    prediction = model.predict([[current, flow]])[0]
    score = round(float(prediction), 1)
 
    # Return JSON reply to frontend
    return jsonify({
        "water_health_score": score,
        "status": "WARNING" if score < 30 else "NORMAL"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
