import os
import io
import logging
import time
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from db import (
    init_db,
    store_prediction,
    store_feedback,
    get_feedback,
    get_all_feedbacks,
    update_feedback,
    delete_feedback,
    get_model_metrics,
)

app = Flask(__name__)

# Request metrics
REQUEST_COUNT = Counter("app_requests_total", "Total requests")
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency in seconds")

# Model performance metrics (post-deployment tracking)
MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total predictions made",
    ["predicted_label"]
)
MODEL_FEEDBACK_TOTAL = Gauge("model_feedback_total", "Total feedback received")
MODEL_FEEDBACK_CORRECT = Gauge("model_feedback_correct_total", "Correct predictions from feedback")
MODEL_FEEDBACK_INCORRECT = Gauge("model_feedback_incorrect_total", "Incorrect predictions from feedback")
MODEL_ACCURACY = Gauge("model_accuracy", "Model accuracy from user feedback (0-1)")
MODEL_PRECISION_CAT = Gauge("model_precision_cat", "Precision for Cat class")
MODEL_PRECISION_DOG = Gauge("model_precision_dog", "Precision for Dog class")
MODEL_RECALL_CAT = Gauge("model_recall_cat", "Recall for Cat class")
MODEL_RECALL_DOG = Gauge("model_recall_dog", "Recall for Dog class")

# Feedback CRUD metrics
FEEDBACK_CRUD_TOTAL = Counter(
    "feedback_crud_operations_total",
    "Total feedback CRUD operations",
    ["operation", "status"],
)

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    latency = time.time() - request.start_time
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(latency)
    return response

@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint. Refreshes model performance gauges from DB."""
    m = get_model_metrics()
    if m:
        MODEL_FEEDBACK_TOTAL.set(m["total_feedback"])
        MODEL_FEEDBACK_CORRECT.set(m["correct"])
        MODEL_FEEDBACK_INCORRECT.set(m["incorrect"])
        MODEL_ACCURACY.set(m["accuracy"])
        MODEL_PRECISION_CAT.set(m["precision_cat"])
        MODEL_PRECISION_DOG.set(m["precision_dog"])
        MODEL_RECALL_CAT.set(m["recall_cat"])
        MODEL_RECALL_DOG.set(m["recall_dog"])
    return Response(generate_latest(), mimetype="text/plain")


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
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    """Prediction endpoint accepting an image and returning probabilities."""
    if model is None:
         return jsonify({"error": "Model not available"}), 400
         
    
        
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

        # Store prediction for performance tracking
        prediction_id = store_prediction(label, dog_prob, cat_prob)
        MODEL_PREDICTIONS_TOTAL.labels(predicted_label=label).inc()

        response_data = {
            "prediction": label,
            "confidence": {
                "Cat": round(cat_prob * 100, 2),
                "Dog": round(dog_prob * 100, 2)
            }
        }
        if prediction_id:
            response_data["prediction_id"] = prediction_id

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Submit user feedback for a prediction (post-deployment model evaluation).
    Expects JSON: { "prediction_id": "uuid", "actual_label": "Cat"|"Dog", "was_correct": true|false }
    """
    if not request.is_json:
        FEEDBACK_CRUD_TOTAL.labels(operation="create", status="error").inc()
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        FEEDBACK_CRUD_TOTAL.labels(operation="create", status="error").inc()
        return jsonify({"error": "Invalid JSON body"}), 400

    prediction_id = data.get("prediction_id")
    actual_label = data.get("actual_label")
    was_correct = data.get("was_correct")

    if not prediction_id:
        FEEDBACK_CRUD_TOTAL.labels(operation="create", status="error").inc()
        return jsonify({"error": "prediction_id is required"}), 400
    if actual_label not in ("Cat", "Dog"):
        FEEDBACK_CRUD_TOTAL.labels(operation="create", status="error").inc()
        return jsonify({"error": "actual_label must be 'Cat' or 'Dog'"}), 400
    if was_correct not in (True, False):
        FEEDBACK_CRUD_TOTAL.labels(operation="create", status="error").inc()
        return jsonify({"error": "was_correct must be true or false"}), 400

    ok = store_feedback(prediction_id, actual_label, was_correct)
    if not ok:
        FEEDBACK_CRUD_TOTAL.labels(operation="create", status="error").inc()
        return jsonify({"error": "Failed to store feedback (DB unavailable or invalid prediction_id)"}), 500

    FEEDBACK_CRUD_TOTAL.labels(operation="create", status="success").inc()
    return jsonify({"status": "ok", "message": "Feedback recorded"}), 200


# --- Feedback CRUD endpoints ---

@app.route('/feedbacks', methods=['GET'])
def list_feedbacks():
    """List all feedbacks with optional pagination. Query params: limit (default 100), offset (default 0)."""
    try:
        limit = min(int(request.args.get("limit", 100)), 500)
    except (TypeError, ValueError):
        limit = 100
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except (TypeError, ValueError):
        offset = 0

    feedbacks = get_all_feedbacks(limit=limit, offset=offset)
    FEEDBACK_CRUD_TOTAL.labels(operation="read", status="success").inc()
    return jsonify({"feedbacks": feedbacks, "count": len(feedbacks)}), 200


@app.route('/feedbacks/<int:feedback_id>', methods=['GET'])
def get_feedback_by_id(feedback_id):
    """Get a single feedback by id."""
    fb = get_feedback(feedback_id)
    if fb is None:
        FEEDBACK_CRUD_TOTAL.labels(operation="read", status="error").inc()
        return jsonify({"error": "Feedback not found"}), 404
    FEEDBACK_CRUD_TOTAL.labels(operation="read", status="success").inc()
    return jsonify(fb), 200


@app.route('/feedbacks/<int:feedback_id>', methods=['PUT', 'PATCH'])
def update_feedback_by_id(feedback_id):
    """
    Update a feedback. Expects JSON: { "actual_label": "Cat"|"Dog"?, "was_correct": true|false? }
    At least one field required.
    """
    if not request.is_json:
        FEEDBACK_CRUD_TOTAL.labels(operation="update", status="error").inc()
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json() or {}
    actual_label = data.get("actual_label")
    was_correct = data.get("was_correct")

    if actual_label is not None and actual_label not in ("Cat", "Dog"):
        FEEDBACK_CRUD_TOTAL.labels(operation="update", status="error").inc()
        return jsonify({"error": "actual_label must be 'Cat' or 'Dog'"}), 400
    if was_correct is not None and was_correct not in (True, False):
        FEEDBACK_CRUD_TOTAL.labels(operation="update", status="error").inc()
        return jsonify({"error": "was_correct must be true or false"}), 400
    if actual_label is None and was_correct is None:
        FEEDBACK_CRUD_TOTAL.labels(operation="update", status="error").inc()
        return jsonify({"error": "At least one of actual_label or was_correct is required"}), 400

    ok = update_feedback(feedback_id, actual_label=actual_label, was_correct=was_correct)
    if not ok:
        FEEDBACK_CRUD_TOTAL.labels(operation="update", status="error").inc()
        return jsonify({"error": "Feedback not found or update failed"}), 404

    FEEDBACK_CRUD_TOTAL.labels(operation="update", status="success").inc()
    return jsonify({"status": "ok", "message": "Feedback updated"}), 200


@app.route('/feedbacks/<int:feedback_id>', methods=['DELETE'])
def delete_feedback_by_id(feedback_id):
    """Delete a feedback by id."""
    ok = delete_feedback(feedback_id)
    if not ok:
        FEEDBACK_CRUD_TOTAL.labels(operation="delete", status="error").inc()
        return jsonify({"error": "Feedback not found"}), 404
    FEEDBACK_CRUD_TOTAL.labels(operation="delete", status="success").inc()
    return jsonify({"status": "ok", "message": "Feedback deleted"}), 200


# Initialize database on startup (runs on import, including under gunicorn)
if init_db():
    print("Database initialized for model performance tracking.")
else:
    print("Database not available. Model metrics will not be persisted.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)