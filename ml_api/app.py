# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import gradio as gr
# from tensorflow import keras
# from tensorflow.keras.preprocessing.text import tokenizer_from_json

# # Load tokenizer correctly from JSON string
# with open("tokenizer.json", "r", encoding="utf-8") as f:
#     tokenizer_json = f.read()
# tokenizer = tokenizer_from_json(tokenizer_json)

# # Load model
# model = keras.models.load_model("BestModel.h5")

# def predict_sentiment(text):
#     if not text.strip():
#         return "Error: Empty text"

#     text_vec = tokenizer.texts_to_matrix([text], mode="binary")
#     prediction = float(model.predict(text_vec)[0][0])
#     sentiment = "Positive" if prediction >= 0.5 else "Negative"

#     return f"{sentiment} (confidence: {round(prediction,3)})"

# gr.Interface(
#     fn=predict_sentiment,
#     inputs=gr.Textbox(lines=3, placeholder="Enter text"),
#     outputs="text",
#     title="Sentiment Analysis (DL + CNN)",
#     description="Deep Learning based Sentiment Analyzer deployed on Hugging Face"
# ).launch()
# launch()


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


from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- CONFIG ----------------
MODEL_PATH = "best_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAXLEN_PATH = "maxlength.pkl"
# ----------------------------------------

app = Flask(__name__)
CORS(app)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Load max_length
with open(MAXLEN_PATH, "rb") as f:
    max_length = pickle.load(f)

# Load trained model
model = load_model(MODEL_PATH)

# ---------------- ROUTES ----------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Sentiment API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Tokenize & pad using saved max_length
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding="post")

    # Prediction
    probability = model.predict(padded)[0][0]
    sentiment = "Positive" if probability >= 0.5 else "Negative"

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": float(probability)
    })

# ---------------- MAIN ----------------

if __name__ == "__main__":
    app.run(debug=True)
