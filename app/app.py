from flask import Flask, render_template, request
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

# Check allowed file types
def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Clean and recreate upload folder
def resetUploadFolder():
    folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

@app.context_processor
def inject_now():
    return {'current_year': datetime.now().year}

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    message = ""

    if request.method == "POST":
        # Option 1: Use default dataset
        if 'default' in request.form:
            resetUploadFolder()
            processDefaultDataset()
            testWithDefaultDataset()
            message = "Default dataset was processed."

        # Option 2: Handle uploaded file or zip
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowedFile(file.filename):
                resetUploadFolder()
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)

                if filename.endswith(".zip"):
                    with zipfile.ZipFile(save_path, 'r') as zip_ref:
                        zip_ref.extractall(app.config['UPLOAD_FOLDER'])

                    processFolder(app.config['UPLOAD_FOLDER'])
                    testWithCustomDataset()
                    message = "Images from ZIP file were processed."

                else:
                    processImage(save_path)
                    testWithCustomDataset()
                    message = "Single image was processed."
            else:
                message = "Unsupported file type."

    return render_template("index.html", message=message)

# Start the server
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
