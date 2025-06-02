import os
import argparse
from preprocessing.preprocessImages import *
from nn.resnet_NN_test import testWithDefaultDataset, testWithCustomDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image dataset preprocessing tool")
    
    parser.add_argument(
        "-c", "--create",
        action="store_true",
        help="Create a new dataset with training and test images"
    )

    parser.add_argument(
        "-p", "--path",
        type=str,
        help="Path for individual image processing (resize and pixelate)"
    )

    parser.add_argument(
        "-b", "--bulk",
        type=str,
        help="Folder path for bulk image processing"
    )

    args = parser.parse_args()

    if args.create:
        createDataset()
        testWithDefaultDataset()

    elif args.path and os.path.isfile(args.path):
        processIndividualImage(args.path)
        testWithCustomDataset()

    elif args.bulk and os.path.isdir(args.bulk):
        processFolderImages(args.bulk)
        testWithCustomDataset()

    else:
        print("\nWarning: No valid action or path specified!\n" +
            "\nUse:\n\t-c to create dataset\n\t" + 
            "-p <imagePath> for single image processing\n\t" + 
            "-b <folderPath> for bulk image processing\n")
