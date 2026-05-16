from flask import Flask, flash, request, redirect, render_template
import os
import tensorflow as tf
import keras.utils as image
from keras.applications.densenet import preprocess_input
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.secret_key = "secret key"

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = tf.keras.models.load_model('model.h5', compile=False)
print('Model loaded successfully')

DISEASE_CLASSES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def model_predict(filename):
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)


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
    if not (file and allowed_file(file.filename)):
        flash('Allowed image types are: png, jpg, jpeg, gif')
        return redirect(request.url)

    # Always clear previous upload so the folder has exactly one file
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    flash('Image successfully uploaded')
    return render_template('index.html', filename=filename)


@app.route('/predict', methods=['GET'])
def predict():
    files = os.listdir(UPLOAD_FOLDER)
    if not files:
        flash('Please upload an image first')
        return redirect('/')

    filename = files[0]
    preds = model_predict(filename)
    ind = np.argmax(preds[0])
    result = DISEASE_CLASSES[ind]
    confidence = round(float(preds[0][ind]) * 100, 1)
    img = UPLOAD_FOLDER + filename
    return render_template('predict.html', result=result, confidence=confidence, img=img)


if __name__ == "__main__":
    app.run(debug=True)
