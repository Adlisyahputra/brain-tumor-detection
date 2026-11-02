import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # maks 5 MB

# Model dimuat sekali saat server menyala
model = load_model('models/brain_tumor_vgg16_small.keras', compile=False)

# Urutan HARUS sama dengan sumbu confusion matrix di notebook
class_labels = ['meningioma', 'pituitary', 'glioma', 'notumor']
IMAGE_SIZE = 128
ALLOWED = {'png', 'jpg', 'jpeg'}


def predict_tumor(file_bytes):
    img = load_img(io.BytesIO(file_bytes), target_size=(IMAGE_SIZE, IMAGE_SIZE))
    x = img_to_array(img) / 255.0          # normalisasi, sama dengan training
    x = np.expand_dims(x, axis=0)          # bentuk (1, 128, 128, 3)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        'prediction': class_labels[idx],
        'has_tumor': class_labels[idx] != 'notumor',
        'confidence': round(float(probs[idx]) * 100, 2),
        'probabilities': {n: round(float(p) * 100, 2)
                          for n, p in zip(class_labels, probs)},
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify(error='Tidak ada file yang dipilih.'), 400
    if file.filename.rsplit('.', 1)[-1].lower() not in ALLOWED:
        return jsonify(error='Format harus PNG, JPG, atau JPEG.'), 400
    try:
        return jsonify(predict_tumor(file.read()))
    except Exception:
        return jsonify(error='Gambar tidak bisa diproses.'), 400


if __name__ == '__main__':
    app.run(debug=True)  # debug hanya untuk pengembangan lokal
