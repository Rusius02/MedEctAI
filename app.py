from flask import Flask, render_template, request
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from model import analyze_data, train_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            analysis = analyze_data(df)
            return render_template('index.html', data=analysis, filename=file.filename)
    return render_template('index.html', data=None)


@app.route('/train', methods=['POST'])
def train():
    filename = request.form['filename']
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(filepath)
    report, accuracy = train_model(df)
    return render_template('index.html', data={'accuracy': accuracy, 'report': report}, filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
