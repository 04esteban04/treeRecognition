import os
import cv2
import numpy as np
import shutil

# Create the directory structure for the target dataset (dataset2/train and dataset2/test)
def createDirectoryStructure(baseDir):
    
    print("\nCreating directories for dataset...\n")

    # If the base directory already exists, delete it to start fresh
    if os.path.exists(baseDir):
        print(f"\tDirectory '{baseDir}' already exists. Removing it...")
        shutil.rmtree(baseDir)

    # Create base directory and subdirectories for train and test
    for subDir in ['train', 'test']:
        targetPath = os.path.join(baseDir, subDir)
        os.makedirs(targetPath, exist_ok=True)
        print(f"\tCreated directory: {targetPath}")
    
    print("\nDirectories successfully created!\n")

# Copy all class subfolders from the source dataset into the destination folder
def copyClassDirectories(sourceDir, destinationDir):

    print("\nCopying directories ...\n")

    for className in os.listdir(sourceDir):
        sourceClassDir = os.path.join(sourceDir, className)
        destinationClassDir = os.path.join(destinationDir, className)
        
        # Only copy if it's a directory
        if os.path.isdir(sourceClassDir):
            shutil.copytree(sourceClassDir, destinationClassDir, dirs_exist_ok=True)
            print(f"\tCopied directory: {sourceClassDir} -> {destinationClassDir}")

    print("\nFinished copying directories!\n")

# Process all images in the given folder:
# - In 'train': keep only the resized version
# - In 'test': keep only the pixelated version
def processImagesToCreateDataset(folderPath, pixelateOnly=False):
    
    print(f"\nProcessing and deleting original images in {folderPath} ...\n")
    
    for root, _, files in os.walk(folderPath):
        for fileName in files:
            if fileName.lower().endswith(('.jpg', '.jpeg', '.png')):
                inputPath = os.path.join(root, fileName)

                try:
                    # Load the image
                    image = loadImage(inputPath)

                    # Apply filters
                    resizedImage = resizeImage(image)
                    blurredImage = blurImage(resizedImage)  # Optional, not saved
                    pixelatedImage = pixelateImage(resizedImage)

                    # Get file name and extension
                    nameOnly, extension = os.path.splitext(fileName)

                    # Determine output path based on mode
                    if not pixelateOnly:
                        outputPath = os.path.join(root, f"{nameOnly}_resized{extension}")
                        cv2.imwrite(outputPath, resizedImage)
                    else:
                        outputPath = os.path.join(root, f"{nameOnly}_pixelate{extension}")
                        cv2.imwrite(outputPath, pixelatedImage)

                    # Delete the original image
                    os.remove(inputPath)

                    print(f"\t{inputPath}")

                except Exception as error:
                    print(f"Error processing {inputPath}: {error}")
    
    print(f"\nFinished processing and deleting original images in {folderPath}!\n")
    print("=========================================================\n")

# Main function to create the dataset
def createDataset():
    # Define source and destination dataset paths
    baseDir = os.path.dirname(os.path.abspath(__file__))
    sourceDataset = os.path.join(baseDir, "dataset")
    destinationDataset = os.path.join(baseDir, "dataset2")

    # Step 1: Create directory structure for destination dataset (train/test)
    createDirectoryStructure(destinationDataset)

    # Step 2: Copy all class folders from source to destination/train
    copyClassDirectories(
        sourceDir=sourceDataset,
        destinationDir=os.path.join(destinationDataset, "train")
    )

    # Step 3: Process images in dataset2/train (keep only resized images)
    processImagesToCreateDataset(
        folderPath=os.path.join(destinationDataset, "train"),
        pixelateOnly=False
    )

    # Step 4: Copy all class folders from source to destination/test
    copyClassDirectories(
        sourceDir=sourceDataset,
        destinationDir=os.path.join(destinationDataset, "test")
    )

    # Step 5: Process images in dataset2/test (keep only pixelated images)
    processImagesToCreateDataset(
        folderPath=os.path.join(destinationDataset, "test"),
        pixelateOnly=True
    )

# Process a single image 
# - Resize, pixelate and save the image
def processIndividualImage(imagePath):

    print("\n*** Individual image processing mode selected! ***" +
        f"\n\nProvided path: \n\t{imagePath}\n")
                
    # Load the image
    image = loadImage(imagePath)

    # Apply filters
    resizedImage = resizeImage(image)
    pixelatedImage = pixelateImage(resizedImage)

    # Construct output file path
    baseDir = os.path.dirname(os.path.abspath(__file__))
    destinationDir = os.path.join(baseDir, "outputPreprocess")
    outputDir = createOutputDirectory(destinationDir)
    outputPath = os.path.join(outputDir, os.path.basename(imagePath))

    # Save the pixelated image created
    cv2.imwrite(outputPath, pixelatedImage)
    print(f"Input image saved to: \n\t{outputPath}\n")

# Process all images in the given folder
# - Resize, pixelate and save the images
def processFolderImages(folderPath):
    
    print("\n*** Bulk image processing mode selected! ***" +
          f"\n\nProvided folder: \n\t{folderPath}\n")

    validExtensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    all_files = os.listdir(folderPath)
    image_files = [f for f in all_files if f.lower().endswith(validExtensions)]

    if not image_files:
        print("No valid images found in the folder.")
        return

    baseDir = os.path.dirname(os.path.abspath(__file__))
    destinationDir = os.path.join(baseDir, "outputPreprocess")
    outputDir = createOutputDirectory(destinationDir)

    print("Input images in folder saved to:")
    for filename in image_files:
        imagePath = os.path.join(folderPath, filename)
        image = loadImage(imagePath)
        resizedImage = resizeImage(image)
        pixelatedImage = pixelateImage(resizedImage)

        outputPath = os.path.join(outputDir, filename)
        cv2.imwrite(outputPath, pixelatedImage)
        print(f"\t{outputPath}")

    print("\nBulk processing completed!\n")

# Create the output directory if it doesn't exist
def createOutputDirectory(outputDir):

    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    
    os.makedirs(outputDir, exist_ok=True)

    return outputDir

# Load an image from the given path
def loadImage(imagePath):

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

# Resize the given image to a fixed size
def resizeImage(image, size=(228, 228), interpolation=cv2.INTER_AREA):
    return cv2.resize(image, size, interpolation=interpolation)

# Apply Gaussian blur to the given image
def blurImage(image, kernel=(5, 5)):
    return cv2.GaussianBlur(image, kernel, 0)

# Apply pixelation effect to the given image
def pixelateImage(image, pixelSize=3):
    height, width = image.shape[:2]
    
    # Shrink image
    temp = cv2.resize(image, (width // pixelSize, height // pixelSize), interpolation=cv2.INTER_LINEAR)
    
    # Resize back to original size using nearest neighbor (blocky effect)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
