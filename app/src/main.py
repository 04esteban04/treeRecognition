from preprocessing.preprocess import *
from reconstruction.reconstruct import *

if __name__ == "__main__":
    
    tensorFromImage = preprocessImage('app/src/assets/test_image.jpg')

    print(tensorFromImage)
    print(tensorFromImage[0].shape, "\n")

    imageFromTensor = reconstructImage(tensorFromImage[0], tensorFromImage[1])

    print(imageFromTensor)
    print(imageFromTensor[0].shape)