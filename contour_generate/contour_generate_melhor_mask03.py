import cv2
import numpy as np
import pickle
import os
import json
from rembg import remove
from PIL import Image

def load_image_and_remove_bg(image_path):
    with open(image_path, 'rb') as file:
        input_image = file.read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    return img[:, :, :3] if img is not None else None

def create_mask(img):
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)

def extract_contours(mask):
    """Extrai contornos da máscara e os retorna em forma vetorizada."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def save_contours(contours, directory="contours"):
    """Salva os contornos em formato JSON."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    contours_data = [contour.tolist() for contour in contours]  
    filename = os.path.join(directory, "contours.json")
    with open(filename, 'w') as f:
        json.dump(contours_data, f)
    print(f"Contornos salvos em {filename}")

def main():
    image_path = input("Digite o caminho da imagem: ")
    image = load_image_and_remove_bg(image_path)
    if image is None:
        print("Erro ao processar a imagem.")
        return
    mask = create_mask(image)
    contours = extract_contours(mask)
    
    save_contours(contours)  # Salva os contornos vetorizados

if __name__ == "__main__":
    main()
