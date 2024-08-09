from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import cv2
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.losses import mean_absolute_error

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Define the mae function
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Load the pre-trained model
g_model = load_model('models/g_model.h5', custom_objects={'InstanceNormalization': InstanceNormalization, 'mae': mae})

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(256, 256))
    img = img_to_array(img)
    norm_img = (img.copy() - 127.5) / 127.5
    return norm_img

def generate_image(input_image):
    g_img = g_model.predict(np.expand_dims(input_image, 0))[0]
    g_img = (g_img * 127.5) + 127.5
    g_img = g_img.astype(np.uint8)
    return g_img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            input_image = preprocess_image(file_path)
            generated_image = generate_image(input_image)

            generated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_' + file.filename)
            cv2.imwrite(generated_image_path, cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))

            return render_template('index.html', uploaded_image=file_path, generated_image=generated_image_path)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
