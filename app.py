import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

# Configuration
MODEL_PATH = './best_model.h5'
TARGET_SIZE = (224, 224) # Update this to match your model's expected input shape

# Load the model globally so it's ready in memory
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Warning: Could not load model. {e}")

def preprocess_image(image_bytes):
    """Prepares the uploaded image for the model."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(TARGET_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Standard normalization (adjust if your model uses different scaling)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to ensure API and model are operational."""
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "Model failed to load"}), 503
    return jsonify({"status": "healthy", "model": MODEL_PATH}), 200

@app.route("/predict", methods=["POST"])
def predict():

    # 1️⃣ Validate request FIRST
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 2️⃣ Then check model availability
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    # 3️⃣ Continue processing
    try:
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)[0][0]

        label = "Dog" if prediction > 0.5 else "Cat"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception:
        return jsonify({"error": "Invalid image"}), 400

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=8000, debug=True)