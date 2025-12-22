from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# Load tokenizer
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Load trained DL model
model = keras.models.load_model("BestModel.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if text.strip() == "":
        return jsonify({"error": "Empty text"}), 400

    # Convert text to model input
    text_vec = tokenizer.texts_to_matrix([text], mode="binary")

    prediction = model.predict(text_vec)[0][0]
    sentiment = 1 if prediction >= 0.5 else 0
    
    print("Request received")   # DEBUG
    data = request.get_json()
    print(data) 
    return jsonify({
        "sentiment": sentiment,
        "confidence": round(float(prediction), 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


