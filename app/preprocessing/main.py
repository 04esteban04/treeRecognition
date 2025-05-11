import os
import argparse
from preprocessImages import *

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

    args = parser.parse_args()

    if args.create:
        createDataset()

    elif args.path and os.path.exists(args.path):
        processIndividualImage(args.path)

    else:
        print("\nWarning: No action specified!" + 
            "\nPlease use -c to create dataset or -p <path> for custom processing.\n")
