# app.py

import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
from image_search import search_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load features and paths once
features = np.load("features.npy")
paths = np.load("paths.npy", allow_pickle=True)

# Image preprocessing function
def extract_feature(img_path):
    img = Image.open(img_path).resize((224, 224)).convert('RGB')
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 3)
    # Simple feature: flatten (replace with better model later)
    return img.flatten()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        query_feature = extract_feature(filepath)
        results = search_image(query_feature)

        return render_template('index.html', query_path=filepath, results=results)

    return render_template('index.html', query_path=None, results=None)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
