from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from preprocessing.preprocessImages import processImage, processFolder, processDefaultDataset
from nn.resnet_NN_test import testWithDefaultDataset, testWithCustomDataset
from datetime import datetime
import os
import shutil
import zipfile
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp', 'zip'}
TEMPLATE_DIR = os.path.abspath("UI/templates")
STATIC_DIR = os.path.abspath("UI/static")

# Initialize Flask app
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check file extensions
def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create upload folder if it doesn't exist
def resetUploadFolder():
    folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Context processor to inject current year into templates
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

# Create report pdf
@app.route('/export-pdf')
def export_pdf():
    return generate_prediction_pdf()

# Generate PDF with predictions and metrics
def generate_prediction_pdf():
    try:
        # === Setup Paths ===
        base_dir = os.path.dirname(os.path.abspath(__file__))
        paths = {
            "csv": os.path.join(base_dir, "nn", "outputModel", "test", "allPredictions_customDataset.csv"),
            "images": os.path.join(base_dir, "preprocessing", "outputPreprocess"),
            "metrics_img": os.path.join(base_dir, "nn", "outputModel", "train", "trainingMetrics_resnetCustomDataset.png"),
            "output_pdf": os.path.join(base_dir, "nn", "outputModel", "test", "prediction_results.pdf"),
            "confusion_csv": os.path.join(base_dir, "nn", "outputModel", "train", "confusion_matrix.csv"),
            "classification_csv": os.path.join(base_dir, "nn", "outputModel", "train", "classification_report.csv")
        }

        if not os.path.exists(paths["csv"]):
            return {"error": "Prediction CSV file not found."}, 404

        df = pd.read_csv(paths["csv"])

        # === Initialize PDF ===
        doc = SimpleDocTemplate(paths["output_pdf"], pagesize=letter, rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
        styles = getSampleStyleSheet()
        elements = []

        # === PDF Title ===
        elements += [Paragraph("Prediction Results Summary", styles['Title']), Spacer(1, 12)]

        # === Predictions Table ===

        # Define symbols
        check_symbol = '✓'
        cross_symbol = '✗'

        centered_style = ParagraphStyle(
            'centered_style',
            parent=styles['Normal'],
            alignment=1,  # 0=left, 1=center, 2=right, 4=justify
        )

        table_data = [["ID", "Image", "Real value", "Predicted value", "Accuracy (%)", "Predicted correctly?"]]
        for _, row in df.iterrows():
            img_path = os.path.join(paths["images"], row['Image name'])
            im = Image(img_path, width=50, height=50) if os.path.exists(img_path) else Paragraph("Image not found", styles["Normal"])
            prediction_correct = str(row["Is prediction correct?"]).strip().lower() == '✔️'

            if prediction_correct:
                prediction_result = Paragraph('<font color="green"><b>%s</b></font>' % check_symbol, centered_style)
            else:
                prediction_result = Paragraph('<font color="red"><b>%s</b></font>' % cross_symbol, centered_style)
            
            table_data.append([
                str(row["Idx"] + 1),
                im,
                row["Real value"],
                row["Predicted"],
                str(row["Prob (%)"]),
                prediction_result
            ])

        table = Table(table_data, colWidths=[40, 60, 120, 120, 60, 80])
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ])

        # Zebra striping
        for i in range(1, len(table_data)):
            bg_color = colors.whitesmoke if i % 2 == 0 else colors.lightgrey
            table_style.add('BACKGROUND', (0, i), (-1, i), bg_color)

        table.setStyle(table_style)
        elements += [table, Spacer(1, 24), PageBreak()]

        # === NN Metrics Section ===
        elements += [Paragraph("NN Model Metrics", styles['Title']), Spacer(1, 12)]
        if os.path.exists(paths["metrics_img"]):
            elements.append(Image(paths["metrics_img"], width=550, height=225))
        else:
            elements.append(Paragraph("Metrics image not found.", styles["Normal"]))

        # === Confusion Matrix Table ===
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("Confusion Matrix", styles['Title']))
        if os.path.exists(paths["confusion_csv"]):
            cm_df = pd.read_csv(paths["confusion_csv"], index_col=0)
            cm_data = [[""] + [col.replace(" (", "\n(") for col in cm_df.columns.tolist()]]
            for idx, row in zip(cm_df.index, cm_df.values.tolist()):
                cm_data.append([idx.replace(" (", "\n(")] + row)

            cm_table = Table(cm_data, colWidths=90, rowHeights=[25] * len(cm_data))
            cm_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d9ead3')),
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f4cccc')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]))
            elements.append(cm_table)
        else:
            elements.append(Paragraph("Confusion matrix CSV not found.", styles['Normal']))

        # === Classification Report Table ===
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("Training Classification", styles['Title']))
        if os.path.exists(paths["classification_csv"]):
            cr_df = pd.read_csv(paths["classification_csv"], index_col=0)
            cr_data = [[""] + [col.replace(" (", "\n(") for col in cr_df.columns]]
            for idx, row in zip(cr_df.index, cr_df.values.tolist()):
                cr_data.append([idx.replace(" (", "\n(")] + row)

            cr_table = Table(cr_data, colWidths=90, rowHeights=[15] + [25] * (len(cr_data) - 1))
            cr_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#cfe2f3')),
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#d9d2e9')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]))
            elements.append(cr_table)
        else:
            elements.append(Paragraph("Training classification report CSV not found.", styles['Normal']))

        # === Finalize PDF ===
        doc.build(elements)
        return send_file(paths["output_pdf"], as_attachment=True)

    except Exception as e:
        return {"error": f"Failed to generate PDF: {str(e)}"}, 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
