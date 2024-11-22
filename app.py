from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import tempfile

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load pretrained models
rfc_model = joblib.load('rfc_trained_model.joblib')
svm_model = joblib.load('svm_trained_model.joblib')
knn_model = joblib.load('knn_trained_model.joblib')
cnn_model = load_model('cnn_parkinsons_model.h5')

# Feature extraction function
def extract_features(audio_path):
    x, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=22).T, axis=0)
    return mfcc.reshape(1, -1)

# Route for file upload and classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file:
        return jsonify({"error": "File not found"}), 400

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        file.save(temp_file.name)
        features = extract_features(temp_file.name)

        # Predict using all models
        rfc_prediction = rfc_model.predict(features)[0]
        svm_prediction = svm_model.predict(features)[0]
        knn_prediction = knn_model.predict(features)[0]
        cnn_prediction = (cnn_model.predict(features.reshape(1, -1, 1)) > 0.5).astype("int32")[0][0]
        cnn_prediction_label = 'PwPD' if cnn_prediction == 1 else 'HC'

        return jsonify({
            "rfc_prediction": rfc_prediction,
            "svm_prediction": svm_prediction,
            "knn_prediction": knn_prediction,
            "cnn_prediction": cnn_prediction_label,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(temp_file.name)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
