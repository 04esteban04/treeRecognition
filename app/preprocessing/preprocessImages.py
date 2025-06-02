import os
import cv2
import numpy as np
import shutil

"""
Image Preprocessing Module
This module provides functions to preprocess images for tree species classification using
a default dataset, a provided image path (individual processing) or folder path (bulk processing).
"""

# Create a default dataset with training and test images
def processDefaultDataset():
    
    print(f"\n{'='*60}" + "\n\tDefault image processing mode selected!\n" + "="*60 + "\n")

    baseDir = os.path.dirname(os.path.abspath(__file__))
    sourceDataset = os.path.join(baseDir, "dataset")
    destinationDataset = os.path.join(baseDir, "dataset2")

    # Reset base directory and subdirectories
    auxResetDirectory(destinationDataset)
    for subDir in ['train', 'test']:
        auxResetDirectory(os.path.join(destinationDataset, subDir))

    # Copy and process default dataset images
    for subDir, pixelateOnly in [('train', False), ('test', True)]:
        
        print(f"Preprocessing {subDir} images...")

        targetDir = os.path.join(destinationDataset, subDir)
        
        auxCopyDefaultDatasetFolders(sourceDataset, targetDir)

        mode = 'pixelate' if pixelateOnly else 'resize'
        auxProcessDefaultImages(targetDir, mode)

    print("\nDefault preprocessing completed!\n" + f"\n{'='*60}")

# Process a single image in the provided path
def processImage(imagePath):
    
    print(f"\n{'='*60}" + "\n\tIndividual image processing mode selected!\n" + "="*60 +
        f"\n\nProvided path: \n\t{imagePath}\n")

    baseDir = os.path.dirname(os.path.abspath(__file__))
    outputDir = auxCreateOutputDirectory(os.path.join(baseDir, "outputPreprocess"))
    outputPath = auxProcessSingleImage(imagePath, outputDir, mode='pixelate')
    
    print("Individual preprocessing completed!\n" + f"\n{'='*60} \n")

# Process all images in the provided folder
def processFolder(folderPath):
    
    print(f"\n{'='*60}" + "\n\tBulk image processing mode selected!\n" + "="*60 +
          f"\n\nProvided folder: \n\t{folderPath}\n")

    validExtensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    imageFiles = [f for f in os.listdir(folderPath) if f.lower().endswith(validExtensions)]

    if not imageFiles:
        print("No valid images found in the folder.")
        return

    baseDir = os.path.dirname(os.path.abspath(__file__))
    outputDir = auxCreateOutputDirectory(os.path.join(baseDir, "outputPreprocess"))

    print("Input images saved to:")
    for fileName in imageFiles:
        try:
            inputPath = os.path.join(folderPath, fileName)
            outputPath = auxProcessSingleImage(inputPath, outputDir, mode='pixelate')
            print(f"\t{outputPath}")
        except Exception as e:
            print(f"Error processing {fileName}: {e}")

    print("\nBulk preprocessing completed!\n" + f"\n{'='*60} \n")

"""
Auxiliary functions for preprocessing tasks such as loading, resizing, blurring, pixelating, etc.
"""

# Auxiliary function to create the output directory if it doesn't exist
def auxCreateOutputDirectory(outputDir):

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    
    os.makedirs(outputDir, exist_ok=True)
    return outputDir

# Auxiliary function to load an image from the given path
def auxLoadImage(imagePath):
    validExtensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # Check if file extension is valid
    if not imagePath.lower().endswith(validExtensions):
        print(f"*** Error: Invalid image file path, please enter a valid image path! ***\n")
        exit(1)

    image = cv2.imread(imagePath)
    if image is None:
        print(f"*** Error: Could not load image from {imagePath} ***")
        exit(1)

    return image

# Auxiliary function to resize the given image to a fixed size
def auxResizeImage(image, size=(228, 228), interpolation=cv2.INTER_AREA):
    return cv2.resize(image, size, interpolation=interpolation)

# Auxiliary function to apply Gaussian blur to the given image
def auxBlurImage(image, kernel=(5, 5)):
    return cv2.GaussianBlur(image, kernel, 0)

# Auxiliary function to apply pixelation effect to the given image
def auxPixelateImage(image, pixelSize=3):
    height, width = image.shape[:2]
    temp = cv2.resize(image, (width // pixelSize, height // pixelSize), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

# Auxiliary function to process a single image and save it to the output directory
def auxProcessSingleImage(inputPath, outputDir, mode='pixelate'):
    image = auxLoadImage(inputPath)
    resized = auxResizeImage(image)

    if mode == 'resize':
        output = resized
        suffix = '_resized'
    elif mode == 'pixelate':
        output = auxPixelateImage(resized)
        suffix = '_pixelate'
    else:
        raise ValueError("Unsupported mode")

    # Get file name and extension
    name, ext = os.path.splitext(os.path.basename(inputPath))
    outputPath = os.path.join(outputDir, f"{name}{suffix}{ext}")
    cv2.imwrite(outputPath, output)

    return outputPath

# Auxiliary function to reset directories
def auxResetDirectory(path):
    
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.makedirs(path, exist_ok=True)

# Auxiliary function to copy default dataset folders
def auxCopyDefaultDatasetFolders(src, dst):
    
    for className in os.listdir(src):
    
        srcClass = os.path.join(src, className)
        dstClass = os.path.join(dst, className)
    
        if os.path.isdir(srcClass):
            shutil.copytree(srcClass, dstClass, dirs_exist_ok=True)

# Auxiliary function to process images in a folder
def auxProcessDefaultImages(folder, mode):
    
    for root, _, files in os.walk(folder):
        
        for fileName in files:
            
            if fileName.lower().endswith(('.jpg', '.jpeg', '.png')):
                
                inputPath = os.path.join(root, fileName)
                
                try:
                    auxProcessSingleImage(inputPath, root, mode)
                    os.remove(inputPath)
                
                except Exception as error:
                    print(f"Error processing {inputPath}: {error}")
