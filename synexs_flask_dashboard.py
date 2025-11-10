import os
import logging
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load the pre-trained model
model_path = os.getenv('MODEL_PATH', 'path/to/model.h5')
try:
    model = tf.keras.models.load_model(model_path)
except (OSError, IOError) as e:
    logging.error(f"Error loading the model: {e}")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()
        if 'input' not in data:
            return jsonify({'error': 'Input data not provided'}), 400

        input_data = np.array([data['input']])

        # Check if the input data shape matches the model's input shape
        if input_data.shape[1] != model.input_shape[1]:
            return jsonify({'error': 'Input data shape mismatch'}), 400

        # Make the prediction
        output = model.predict(input_data)

        # Return the prediction as a JSON response
        return jsonify({'output': output.tolist()})

    except (ValueError, Exception) as e:
        logging.error(f"Error making prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True, processes=4)
    except (KeyboardInterrupt, Exception) as e:
        logging.error(f"Error running the server: {e}", exc_info=True)
        exit(1)