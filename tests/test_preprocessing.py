import pytest
import torch
import cv2
import os

from src.preprocessing import (
    preprocessImage, 
    loadImage,
    convertBgrToRgb, 
    resizeImage, 
    normalizeAndConvertToTensor)

@pytest.fixture
def sampleImage():
    return os.path.join('assets', 'test_image.jpeg')
    
def test_loadImage(sampleImage):
    img = loadImage(sampleImage)
    assert img is not None
    assert img.shape[2] == 3

def test_convertBgrToRgb(sampleImage):
    img_bgr = loadImage(sampleImage)
    img_rgb = convertBgrToRgb(img_bgr)
    assert img_rgb.shape == img_bgr.shape
    assert (img_rgb[..., 0] != img_bgr[..., 0]).any()

def test_resizeImage(sampleImage):
    img_bgr = loadImage(sampleImage)
    img_rgb = convertBgrToRgb(img_bgr)
    img_resized = resizeImage(img_rgb)
    assert img_resized.shape == (224, 224, 3)

def test_normalizeAndConvertToTensor(sampleImage):
    img_bgr = loadImage(sampleImage)
    img_rgb = convertBgrToRgb(img_bgr)
    img_resized = resizeImage(img_rgb)
    tensor_img = normalizeAndConvertToTensor(img_resized)
    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (3, 224, 224)

def test_preprocessImage(sampleImage):
    tensor_img, original_size = preprocessImage(sampleImage)
    assert isinstance(tensor_img, torch.Tensor)
    assert len(original_size) == 2