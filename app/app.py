from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from preprocessing.preprocessImages import processImage, processFolder, processDefaultDataset
from nn.resnet_NN_test import testWithDefaultDataset, testWithCustomDataset
from datetime import datetime
import os
import shutil
import zipfile

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

# Process default dataset
@app.route("/process/default", methods=["POST"])
def process_default():
    resetUploadFolder()
    processDefaultDataset()
    testWithDefaultDataset()
    return jsonify({"message": "Default dataset was processed."}), 200

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
    return jsonify({"message": "Single image was processed."}), 200

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
    return jsonify({"message": "Images from ZIP were processed."}), 200

# Run the app
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
