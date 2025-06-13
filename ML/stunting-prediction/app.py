from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model dan scaler
model = load_model('Model_stunting_new.keras', compile=False)
scaler = joblib.load('scaler.pkl')

# Mapping hasil prediksi
status_gizi_mapping = {
    0: 'severely stunted',
    1: 'stunted',
    2: 'normal',
    3: 'tinggi'
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data)

        # Konversi jenis kelamin ke numerik
        df['Label_kelamin'] = df['Jenis Kelamin'].map({'laki-laki': 0, 'perempuan': 1})

        # Pilih fitur
        features = df[['Umur (bulan)', 'Tinggi Badan (cm)', 'Label_kelamin']]
        scaled_features = scaler.transform(features)

        # Prediksi
        predictions = model.predict(scaled_features)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = [status_gizi_mapping[i] for i in predicted_classes]

        return jsonify({'predictions': predicted_labels})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
