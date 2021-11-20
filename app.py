# app.py
import os
import numpy as np
from flask import Flask, request, render_template
from fastai.vision.all import *

app = Flask(__name__)
learn = load_learner("./model/sneakers.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # # Get the file from post request
        f = request.files['file']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)

        result = model_predict(file_path)        
        return result
        
    return None

def model_predict(img_path):
    # Make prediction
    img = PILImage.create(img_path)

    pred_result = learn.predict(img)[2].sort(descending=True)
    top_3_pred_probs = pred_result[0][:3]

    # Convert probs to numpy array because I just want the numbers by themselves without 'tensor'
    top_3_pred_probs = top_3_pred_probs.numpy()

    # Round the prediction probabilities from long floats to 2 decimal places
    top_3_pred_probs = [round(i, 2) for i in top_3_pred_probs]

    # Grab the indices so I can use them to lookup the correct value from learn.data.classes
    top_3_pred_class_idxs = pred_result[1][:3]

    # Convert label from 'air_jordan_3' to 'Air Jordan 3' after looking up proper index
    top_3_pred_classes = [learn.dls.vocab[i].replace('_', ' ').title() for i in top_3_pred_class_idxs]

    predictions = list(zip(top_3_pred_classes, top_3_pred_probs))
    texts = [f'{idx+1}: {val[0]}, { "{:.1%}".format(val[1]) }' for idx, val in enumerate(predictions)]

    return texts[0]

if __name__ == '__main__':
    app.run(threaded=True, port=5000)