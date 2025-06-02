# Tree species recognition system🌲

This project is an image processing system designed to classify tree species based on aerial or ground-level images. The goal is to support environmental monitoring and conservation efforts by providing a fast, scalable, and accurate solution for automatic tree species identification.

The system utilizes deep learning techniques, specifically a convolutional neural network based on the ResNet18 architecture, to analyze images of tree canopies. The project includes a full preprocessing pipeline (resizing, normalization, pixelation, etc.), a training and evaluation framework using PyTorch, a graphical user interface for data visualization and a reporting module for exporting classification results.

## Key Features 🔧
- **Automated preprocessing:** Images are resized, normalized, and prepared for model input using OpenCV and PyTorch.

- **Custom dataset handling:** Support for structured class folders and data labeling.

- **Deep learning model:** A ResNet18-based neural network is used for high-performance image classification.

- **Reporting and visualization:** Tools for generating PDF reports (via ReportLab) and rendering classification results in the UI with Flask.

- **Scalability:** Lightweight design with support for default, individual and batch processing.

- **Environmental focus:** Supports low-resource execution and sustainable computing practices.

## Project Structure 📁

```bash
app/
├── main.py                             # Program execution file (handles dataset creation, classification and reporting
├── preprocessing/                      # Preprocessing and dataset management modules
│   ├── dataset/                        # Default source images organized by class
│   ├── datasetBulk/                    # Optional test input folder for batch processing
│   └── preprocessImage.py              # Image processing functions (resize, pixelate, etc.)
├── nn/                                 # Neural network modules
│   ├── outputModel/                    # Stores model training and testing classification results
│   ├── resnet_NN_train.py              # Neural network module with training functions
│   ├── resnet_NN_test.py               # Neural network module with testing functions
│   ├── resnet_NN_randomAccuracy.py     # Neural network module with randomAccuracy (not being in use)
├── forReference/                       # Neural network modules created as a reference
│   ├── baseNN/                         # Stores simple NN model training and testing reference functions
│   ├──── base_NN_test.py 
│   ├──── base_NN_train.py 
│   ├── resnetNN/                       # Stores resnet NN model training and testing reference functions
│   ├──── resnet_NN_test.py 
│   ├──── resnet_NN_train.py 

```

## Use Cases 💡
- Ecological monitoring
- Forestry management
- Biodiversity research
- Environmental impact analysis

## Technologies Used 📦

- Python
- PyTorch
- OpenCV
- Numpy
- Pandas
- Matplotlib
- JSON
- sklearn
- argparse
- ReportLab


## Usage 📄

To get info on how to use it use:
```bash
python main.py -h
```

To create and test with the default dataset use:
```bash
python main.py -c
```

To test with an image path for individual processing use: 
```bash
python main.py -p '<image_path>'
```

To test with a folder path for bulk image processing use:
```bash
python main.py -b '<dir_path>'
```

> [!IMPORTANT]  
> For individual or bulk processing, the input files should follow the convention: <br>
> <div align="center"><code>&lt;name&gt;_&lt;treeSpecies&gt;.&lt;file_extension&gt;</code></div> <br>
> For example: <code>img9_roble.jpg</code>