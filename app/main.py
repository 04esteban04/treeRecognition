import os
import argparse
from preprocessing.preprocessImages import *
from nn.resnet_NN_test import testWithDefaultDataset, testWithCustomDataset

if __name__ == "__main__":
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Tree species classification tool")
    
    parser.add_argument(
        "-c", "--create",
        action="store_true",
        help="Create a default dataset with training and test images"
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
        # Create a new dataset and run tests using the default dataset
        processDefaultDataset()
        testWithDefaultDataset()

    elif args.path and os.path.isfile(args.path):
        # Process a single image in the provided path
        processImage(args.path)
        testWithCustomDataset()

    elif args.bulk and os.path.isdir(args.bulk):
        # Process all images in the provided folder
        processFolder(args.bulk)
        testWithCustomDataset()

    else:
        # Show a warning if no valid argument or path is provided
        print("\n" + parser.description + 
            "\n\nWarning: No valid action or path specified!" +
            "\nUse: -h to get help on how to use the tool\n")
