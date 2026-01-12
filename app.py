from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists("model.pkl"):
            raise FileNotFoundError("model.pkl not found. Train the model first.")
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    return model

@app.route("/predict", methods=["POST"])
def predict():
    features = request.json["features"]
    mdl = load_model()
    prediction = mdl.predict([features])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)