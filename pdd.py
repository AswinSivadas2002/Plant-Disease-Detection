from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import tensorflow as tf
from tensorflow import keras
import keras.utils as image
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
model = tf.keras.models.load_model('PlantDNet.h5', compile=False)
print('model loaded successfully')

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
file_name=''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def model_predict(model):
    filename = os.listdir(UPLOAD_FOLDER)[0]
    img_path=UPLOAD_FOLDER+filename
    print(img_path)
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        flash('Image successfully uploaded and displayed below')
        #file_name=filename
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds = model_predict(model)
    print(preds[0])

    # x = x.reshape([64, 64]);
    disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                     'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                     'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                     'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                     'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
    a = preds[0]
    ind = np.argmax(a)
    print('Prediction:', disease_class[ind])
    result = disease_class[ind]
    print(result)
    img=UPLOAD_FOLDER+os.listdir(UPLOAD_FOLDER)[0]
    return render_template('predict.html',result=result,img=img)

if __name__ == "__main__":
    app.run(debug=True)