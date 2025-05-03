import cv2
import torch
import numpy as np
import os
from torchvision import transforms

# Media y desviación estándar utilizadas en modelos ResNet preentrenados
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

def getInverseNormalizeTransform():
    """Genera una transformación para desnormalizar una imagen procesada para ResNet."""
    return transforms.Normalize(
        mean=[-m/s for m, s in zip(RESNET_MEAN, RESNET_STD)],
        std=[1/s for s in RESNET_STD]
    )

def denormalizeTensor(tensorImg):
    """Aplica la desnormalización al tensor normalizado."""
    invNormalize = getInverseNormalizeTransform()
    return invNormalize(tensorImg)

def tensorToNumpyImage(tensorImg):
    """Convierte un tensor a un arreglo NumPy en el rango [0, 255] como imagen RGB."""
    imgNp = tensorImg.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    imgNp = (imgNp * 255).astype(np.uint8)
    return imgNp

def resizeImageToOriginal(imgNp, originalSize):
    """Redimensiona una imagen NumPy a su tamaño original."""
    return cv2.resize(imgNp, (originalSize[1], originalSize[0]))

def convertRgbToBgr(imgNp):
    """Convierte una imagen RGB en NumPy a formato BGR para OpenCV."""
    return cv2.cvtColor(imgNp, cv2.COLOR_RGB2BGR)

def saveImage(imgBgr, filename='reconstructed_image.png'):
    """Guarda una imagen en el disco."""

    current_path = os.path.dirname(__file__)
    
    src_path = os.path.abspath(os.path.join(current_path, '..'))
    
    output_folder = os.path.join(src_path, 'output')
    os.makedirs(output_folder, exist_ok=True)

    save_path = os.path.join(output_folder, filename)

    success = cv2.imwrite(save_path, imgBgr)

    return success

def reconstructImage(tensorImg, originalSize):
    """
    Reconstruye una imagen RGB a partir de un tensor normalizado (estilo ResNet).
    Retorna una imagen OpenCV (BGR) en resolución original.
    """
    imgDenorm = denormalizeTensor(tensorImg)
    imgNp = tensorToNumpyImage(imgDenorm)
    imgResized = resizeImageToOriginal(imgNp, originalSize)
    imgBgr = convertRgbToBgr(imgResized)
    saveImage(imgBgr)

    return imgBgr