# app.py
import os
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # # Get the file from post request
        # f = request.files['file']

        # # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', f.filename)
        # f.save(file_path)

        # # Make prediction
        # preds = model_predict(file_path)
        return "Nike"
    return None

def model_predict(img_path):
    """
       model_predict will return the preprocessed image
    """
   
    # img = open_image(img_path)
    # pred_class,pred_idx,outputs = learn.predict(img)
    return "Nike"

if __name__ == '__main__':
    app.run(threaded=True, port=5000)