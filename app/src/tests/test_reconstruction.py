import pytest
import torch
import os
import cv2
import numpy as np 

from preprocessing.preprocess import preprocessImage
from reconstruction.reconstruct import (
    denormalizeTensor, 
    tensorToNumpyImage, 
    resizeImageToOriginal, 
    convertRgbToBgr, 
    saveImage, 
    reconstructImage)

@pytest.fixture
def sampleImage():
    current_dir = os.path.dirname(__file__)
    sampleImage = os.path.join(current_dir, "..", "assets", "test_image.jpg")
    return os.path.abspath(sampleImage)

def test_denormalizeTensor(sampleImage):
    tensor_img, _ = preprocessImage(sampleImage)
    img_denorm = denormalizeTensor(tensor_img)
    assert isinstance(img_denorm, torch.Tensor)
    assert img_denorm.shape == (3, 224, 224)

def test_tensorToNumpyImage(sampleImage):
    tensor_img, _ = preprocessImage(sampleImage)
    img_denorm = denormalizeTensor(tensor_img)
    img_np = tensorToNumpyImage(img_denorm)
    assert isinstance(img_np, np.ndarray)
    assert img_np.shape == (224, 224, 3)
    assert img_np.dtype == np.uint8

def test_resizeImageToOriginal(sampleImage):
    tensor_img, original_size = preprocessImage(sampleImage)
    img_denorm = denormalizeTensor(tensor_img)
    img_np = tensorToNumpyImage(img_denorm)
    img_resized = resizeImageToOriginal(img_np, original_size)
    assert img_resized.shape[0] == original_size[0]
    assert img_resized.shape[1] == original_size[1]

def test_convertRgbToBgr(sampleImage):
    tensor_img, _ = preprocessImage(sampleImage)
    img_denorm = denormalizeTensor(tensor_img)
    img_np = tensorToNumpyImage(img_denorm)
    img_bgr = convertRgbToBgr(img_np)
    assert img_bgr.shape == img_np.shape

def test_saveImage(sampleImage):
    tensor_img, original_size = preprocessImage(sampleImage)
    img_denorm = denormalizeTensor(tensor_img)
    img_np = tensorToNumpyImage(img_denorm)
    img_resized = resizeImageToOriginal(img_np, original_size)
    img_bgr = convertRgbToBgr(img_resized)
    success = saveImage(img_bgr, filename="test_output.png")
    assert success, "La imagen no se pudo guardar"

def test_reconstructImage(sampleImage):
    tensor_img, original_size = preprocessImage(sampleImage)
    img_bgr = reconstructImage(tensor_img, original_size)
    assert img_bgr is not None
    assert isinstance(img_bgr, np.ndarray)
    assert img_bgr.shape[2] == 3
