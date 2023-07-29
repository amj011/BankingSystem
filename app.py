# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

def make_prediction(data):
    predictions = model.predict(data)
    return predictions

@app.route('/', methods=['GET'])
def get_request():
    return {'message': 'Hello, this is a sample GET request!'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    predictions = make_prediction(data)
    return jsonify(predictions)
