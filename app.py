import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

labels=['cardboard','glass','metal','paper','plastic','trash']
model = keras.models.load_model("trashwheelgpubackup")

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img)/255.0

    p = model.predict(img[np.newaxis, ...])
    predicted_class = labels[np.argmax(p[0], axis=-1)]

    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
