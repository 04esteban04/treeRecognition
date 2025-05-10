import os
from preprocessImages import (
    createDirectoryStructure,
    copyClassDirectories,
    processImagesInFolder
)

if __name__ == "__main__":
    # Define source and destination dataset paths
    sourceDataset = "dataset"
    destinationDataset = "dataset2"

    # Step 1: Create directory structure for destination dataset (train/test)
    createDirectoryStructure(destinationDataset)

    # Step 2: Copy all class folders from source to destination/train
    copyClassDirectories(
        sourceDir=sourceDataset,
        destinationDir=os.path.join(destinationDataset, "train")
    )

    # Step 3: Process images in dataset2/train (keep only resized images)
    processImagesInFolder(
        folderPath=os.path.join(destinationDataset, "train"),
        pixelateOnly=False
    )

    # Step 4: Copy all class folders from source to destination/test
    copyClassDirectories(
        sourceDir=sourceDataset,
        destinationDir=os.path.join(destinationDataset, "test")
    )

    # Step 5: Process images in dataset2/test (keep only pixelated images)
    processImagesInFolder(
        folderPath=os.path.join(destinationDataset, "test"),
        pixelateOnly=True
    )
