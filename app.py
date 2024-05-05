import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

import cv2
import numpy as np

from PIL import Image

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return processor(images=image, return_tensors="pt")["pixel_values"][0]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_caption(image):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return pred

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/upload', methods=['GET','POST'])
# def upload_image():
#     if request.method == 'POST':
#         image = request.files['image']
#         if image:
#             filename = secure_filename(image.filename)
#             filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             image.save(filepath)
            
#             image = Image.open(filepath).convert("RGB")
#             caption = predict_caption(image)
#             return render_template('predicted_caption.html', image_path=filepath, caption=caption)
#         else:
#             return "No image uploaded"
#     else:
#         return "Invalid request"
@app.route('/upload', methods=['GET','POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            filepath = os.path.join('uploads', filename)
            image.save(filepath)

            image = Image.open(filepath).convert("RGB")
            caption = predict_caption(image)
            return render_template('predicted_caption.html', image_path=filename, caption=caption)
        else:
            return "No image uploaded"
    else:
        return "Invalid request"

@app.route('/uploads/<path:filename>')
def send_uploaded_file(filename):
    return send_from_directory('uploads', filename)


if __name__ == '__main__':
    app.run(debug=True)
