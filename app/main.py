from flask import request, jsonify
from app import app
from app.model.load_model import model
from app.utils.preprocess import preprocess_image
from PIL import Image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    image = Image.open(file)
    image = preprocess_image(image)
    prediction = model.predict(image)
    return jsonify({'prediction': str(prediction)})
