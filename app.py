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

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint accepting an image and returning probabilities."""
    if model is None:
         return jsonify({"error": "Model not available"}), 503
         
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Run inference
        prediction = model.predict(processed_image)[0][0]
        
        # Assuming binary classification where output close to 1 is Dog, and 0 is Cat.
        # Adjust logic if your model outputs a 2-node softmax array instead of a single sigmoid node.
        dog_prob = float(prediction)
        cat_prob = 1.0 - dog_prob
        
        label = 'Dog' if dog_prob > 0.5 else 'Cat'
        
        return jsonify({
            "prediction": label,
            "confidence": {
                "Cat": round(cat_prob * 100, 2),
                "Dog": round(dog_prob * 100, 2)
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=8000, debug=True)