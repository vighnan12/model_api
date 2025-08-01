from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import io
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model
MODEL_PATH = "plant_disease_model_updated.h5"
model = load_model(MODEL_PATH)

# Class labels (potato removed)
class_labels = [
    "Corn___Common_rust",
    "Corn___Gray_leaf_spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Hispa",
    "Wheat___Brown_rust",
    "Wheat___Healthy",
    "Wheat___Yellow_rust"
]

# Preprocess the input image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'fail', 'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'fail', 'error': 'Empty filename'}), 400

    try:
        image_bytes = file.read()
        image = load_img(io.BytesIO(image_bytes), target_size=(224, 224))
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0]
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            'status': 'success',
            'predicted_class': predicted_class,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        return jsonify({'status': 'fail', 'error': str(e)}), 500

# Start app (for local or Render deployment)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
