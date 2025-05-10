import os
import cv2
import numpy as np

def resizeImage(image, size=(228, 228), interpolation=cv2.INTER_AREA):
    return cv2.resize(image, size, interpolation=interpolation)

def blurImage(image, kernel=(5, 5)):
    return cv2.GaussianBlur(image, kernel, 0)

def pixelateImage(image, pixel_size=3):
    h, w = image.shape[:2]
    temp = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def resizeAndFilter(input_path, output_dir, size=(228, 228), blur_kernel=(5, 5)):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {input_path}")

    # Procesamiento
    resized = resizeImage(image, size)
    blurred = blurImage(resized, blur_kernel)
    pixelated = pixelateImage(resized)

    # Extraer nombre base y extensión
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)

    # Construir rutas de salida
    resized_path = os.path.join(output_dir, f"{name}_resized{ext}")
    blur_path = os.path.join(output_dir, f"{name}_blur{ext}")
    pixel_path = os.path.join(output_dir, f"{name}_pixelate{ext}")

    # Guardar imágenes
    #cv2.imwrite(resized_path, resized)
    #cv2.imwrite(blur_path, blurred)
    cv2.imwrite(pixel_path, pixelated)

    print(f"Imágenes guardadas:\n - {resized_path}\n - {pixel_path}")


if __name__ == "__main__":
    resizeAndFilter("img1.jpg", "./")
    resizeAndFilter("img2.jpg", "./")
    resizeAndFilter("img3.jpg", "./")
    resizeAndFilter("img4.jpg", "./")
    resizeAndFilter("img5.jpg", "./")
    resizeAndFilter("img6.jpg", "./")
    resizeAndFilter("img7.jpg", "./")
    resizeAndFilter("img8.jpg", "./")
    resizeAndFilter("img9.jpg", "./")
    resizeAndFilter("img10.jpg", "./")
    resizeAndFilter("img11.jpg", "./")
    resizeAndFilter("img12.jpg", "./")
