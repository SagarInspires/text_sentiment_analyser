import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# Load tokenizer from JSON
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

# Load trained DL model
model = keras.models.load_model("BestModel.h5")

def predict_sentiment(text):
    if text.strip() == "":
        return "Error: Empty text"

    # Convert text to model input
    text_vec = tokenizer.texts_to_matrix([text], mode="binary")

    prediction = model.predict(text_vec)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"

    return f"{sentiment} (confidence: {round(float(prediction),3)})"

gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter text"),
    outputs="text",
    title="Sentiment Analysis (DL + CNN)",
    description="Deep Learning based Sentiment Analyzer deployed on Hugging Face"
).launch()


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pickle
# from tensorflow import keras
# import os

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer
# with open("tokenizer.pickle", "rb") as f:
#     tokenizer = pickle.load(f)

# # Load trained DL model
# model = keras.models.load_model("BestModel.h5")

# # Root endpoint
# @app.route("/")
# def home():
#     return jsonify({"message": "Sentiment Analysis API is live!"})

# # Prediction endpoint
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     text = data.get("text", "")

#     if not text.strip():
#         return jsonify({"error": "Empty text"}), 400

#     # Convert text to model input
#     text_vec = tokenizer.texts_to_matrix([text], mode="binary")

#     # Predict sentiment
#     prediction = float(model.predict(text_vec)[0][0])
#     sentiment = 1 if prediction >= 0.5 else 0

#     return jsonify({
#         "sentiment": sentiment,
#         "confidence": round(float(prediction), 3)
#     })

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))  # Use Render-assigned port
#     app.run(host="0.0.0.0", port=port)

# from flask import Flask, request, jsonify
# import pickle
# import os

# app = Flask(__name__)

# # Load trained Naive Bayes model and CountVectorizer once
# with open('NB_spam_model.pkl', 'rb') as f:
#     clf = pickle.load(f)

# with open('cv.pkl', 'rb') as f:
#     cv = pickle.load(f)

# # Root endpoint
# @app.route("/")
# def home():
#     return jsonify({"message": "Sentiment analyser is live!"})

# # Prediction endpoint
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     text = data.get("text", "")

#     if not text.strip():
#         return jsonify({"error": "Empty text"}), 400

#     # Convert text to model input
#     text_vec = cv.transform([text]).toarray()

#     # Get probability for class 1 
#     prob = float(clf.predict_proba(text_vec)[0][1])

#     # Apply threshold >= 0.5
#     prediction = 1 if prob >= 0.5 else 0

#     return jsonify({
#         "prediction": prediction,           # 0 or 1
#         "confidence": round(prob, 3)       # probability for class 1
#     })

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)
