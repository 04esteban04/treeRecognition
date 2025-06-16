from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from preprocessing.preprocessImages import processImage, processFolder, processDefaultDataset
from nn.resnet_NN_test import testWithDefaultDataset, testWithCustomDataset
from datetime import datetime
import os
import shutil
import zipfile
import pandas as pd

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'zip'}
TEMPLATE_DIR = os.path.abspath("UI/templates")
STATIC_DIR = os.path.abspath("UI/static")

# Initialize Flask app
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resetUploadFolder():
    folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

@app.context_processor
def inject_now():
    return {'current_year': datetime.now().year}

# Load the main UI
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Serve static files
@app.route("/preprocessed_images/<filename>")
def serve_preprocessed_image(filename):
    baseDir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(baseDir, "preprocessing", "outputPreprocess")
    return send_from_directory(folder, filename)

# Process default dataset
@app.route("/process/default", methods=["POST"])
def process_default():
    resetUploadFolder()
    processDefaultDataset()
    testWithDefaultDataset()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "nn", "outputModel", "test", "allPredictions_customDataset.csv")
    
    if not os.path.exists(csv_path):
        return jsonify({
            "error": "Default prediction CSV file not found."
        }), 404

    try:
        df = pd.read_csv(csv_path)
        results = df.to_dict(orient="records")
        return jsonify({
            "message": "Default dataset was processed.",
            "results": results
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Failed to read predictions: {str(e)}"
        }), 500

# Process individual image
@app.route("/process/image", methods=["POST"])
def process_image():
    file = request.files.get('file')

    if not file or not allowedFile(file.filename):
        return jsonify({"error": "Unsupported or missing file."}), 400

    resetUploadFolder()
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    processImage(path)
    testWithCustomDataset()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "nn", "outputModel", "test", "allPredictions_customDataset.csv")
    
    if not os.path.exists(csv_path):
        return jsonify({
            "error": "Prediction CSV file not found."
        }), 404

    try:
        df = pd.read_csv(csv_path)
        results = df.to_dict(orient="records")
        return jsonify({
            "message": "Single image dataset was processed.",
            "results": results
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Failed to read predictions: {str(e)}"
        }), 500
    
    #return jsonify({"message": "Single image was processed."}), 200

# Process batch images from ZIP
@app.route("/process/batch", methods=["POST"])
def process_batch():
    file = request.files.get('file')

    if not file or not allowedFile(file.filename) or not file.filename.endswith(".zip"):
        return jsonify({"error": "Please upload a valid .zip file."}), 400

    resetUploadFolder()
    filename = secure_filename(file.filename)
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(app.config['UPLOAD_FOLDER'])

    processFolder(app.config['UPLOAD_FOLDER'])
    testWithCustomDataset()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "nn", "outputModel", "test", "allPredictions_customDataset.csv")
    
    if not os.path.exists(csv_path):
        return jsonify({
            "error": "Prediction CSV file not found."
        }), 404

    try:
        df = pd.read_csv(csv_path)
        results = df.to_dict(orient="records")
        return jsonify({
            "message": "Batch of images was processed.",
            "results": results
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Failed to read predictions: {str(e)}"
        }), 500

    #return jsonify({"message": "Images from ZIP were processed."}), 200

# Run the app
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
