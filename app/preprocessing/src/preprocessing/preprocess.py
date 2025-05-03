import cv2
import torch
import numpy as np
from torchvision import transforms

# Media y desviación estándar utilizadas en modelos ResNet preentrenados
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

def loadImage(image_path):
    """Carga una imagen desde el disco usando OpenCV."""
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    """Valida que la imagen sea RGB (3 canales)."""
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("La imagen no es de tipo RGB (3 canales).")

    return img

def convertBgrToRgb(img):
    """Convierte una imagen BGR (OpenCV) a RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resizeImage(img, target_size=(224, 224)):
    """Redimensiona una imagen a las dimensiones requeridas."""
    return cv2.resize(img, target_size)

def normalizeAndConvertToTensor(img):
    """Normaliza y convierte la imagen a un tensor."""
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=RESNET_MEAN, std=RESNET_STD)
    ])
    return transform_pipeline(img)

def preprocessImage(image_path):
    """
    Pipeline completo de preprocesamiento: carga, validación, conversión, redimensionado, normalización.
    Retorna: tensor de imagen preprocesada, tamaño original.
    """
    img = loadImage(image_path)
    img_rgb = convertBgrToRgb(img)
    original_size = img_rgb.shape[:2]  # (alto, ancho)
    img_resized = resizeImage(img_rgb)
    tensor_img = normalizeAndConvertToTensor(img_resized)

    return tensor_img, original_size

