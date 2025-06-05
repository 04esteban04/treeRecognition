from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from preprocessing.preprocessImages import processImage, processFolder, processDefaultDataset
from nn.resnet_NN_test import testWithDefaultDataset, testWithCustomDataset
import os
import shutil
import zipfile

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'zip'}
TEMPLATE_DIR = os.path.abspath("UI/templates")

# Flask app setup
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file extension is allowed
def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Reset upload folder by deleting its contents
def resetUploadFolder():
    folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    message = ""

    if request.method == "POST":
        # Option 1: Use the default dataset
        if 'default' in request.form:
            resetUploadFolder()
            processDefaultDataset()
            testWithDefaultDataset()
            message = "Default dataset was processed."

        # Option 2: Process uploaded file
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowedFile(file.filename):
                resetUploadFolder()
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)

                # If the uploaded file is a ZIP archive, then
                # extract the contents directly into the upload folder
                if filename.endswith(".zip"):
                    with zipfile.ZipFile(save_path, 'r') as zip_ref:
                        zip_ref.extractall(app.config['UPLOAD_FOLDER'])

                    processFolder(app.config['UPLOAD_FOLDER'])
                    testWithCustomDataset()
                    message = "Images from ZIP file were processed."

                # If it's a single image, just save the image and process it
                else:
                    processImage(save_path)
                    testWithCustomDataset()
                    message = "Single image was processed."

    return render_template("index.html", message=message)

# Start the server
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
