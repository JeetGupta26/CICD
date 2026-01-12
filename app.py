from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists("model.pkl"):
            return None
        try:
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
        except Exception:
            model = None
    return model

@app.route("/predict", methods=["POST"])
def predict():
    features = request.json["features"]
    mdl = load_model()

    if mdl is None:
        return jsonify({"error": "Model not available"}), 500

    prediction = mdl.predict([features])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)