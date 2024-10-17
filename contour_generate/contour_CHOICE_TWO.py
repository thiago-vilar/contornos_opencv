import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from rembg import remove
import pickle
import os

def remove_background(filepath):
    """Remove o fundo da imagem especificada no caminho do arquivo."""
    input_image = open(filepath, 'rb').read()
    output_image = remove(input_image)
    image_np_array = np.frombuffer(output_image, np.uint8)
    img = cv2.imdecode(image_np_array, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    return img[:, :, :3]

def read_and_convert_image(image_array):
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return img

def create_mask(img):
    """Cria uma máscara binária para a imagem com base em um intervalo de cores."""
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)

def find_centroid(mask):
    """Encontra o centróide da maior região branca na máscara."""
    M = cv2.moments(mask)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return None

def extract_and_draw_contours(img, mask):
    """Extrai contornos da máscara e desenha-os na imagem."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_with_contours = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0, 255), 1)
    return img_with_contours, contours

def crop_to_contours(img, contours):
    """Recorta a imagem para a bounding box mínima que envolve todos os contornos."""
    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        return img[y:y+h, x:x+w]
    return img

def transform_alpha_mask_to_white_background(image):
    """Transforma uma imagem com canal alpha em uma imagem com fundo branco."""
    image_pil = Image.fromarray(image)
    background = Image.new('RGBA', image_pil.size, (255, 255, 255, 255))
    return np.array(Image.alpha_composite(background, image_pil.convert('RGBA')).convert('RGB'))

def translate_to_origin(contours, centroid):
    """Translada contornos para que o centróide seja a origem (0, 0)."""
    return [contour - centroid for contour in contours]

def save_contours(contours, centroid):
    """Salva os contornos detectados transladados para que o centróide esteja na origem."""
    directory = "contours_by_mask"
    os.makedirs(directory, exist_ok=True)
    translated_contours = translate_to_origin(contours, centroid)
    for i, contour in enumerate(translated_contours):
        filename_pkl = os.path.join(directory, f"contour{i}.pkl")
        with open(filename_pkl, 'wb') as f:
            pickle.dump(contour.reshape(-1, 2), f)

def main():
    filepath = input("Digite o caminho da imagem: ")
    img_no_bg = remove_background(filepath)
    mask = create_mask(img_no_bg)
    img_with_contours, contours = extract_and_draw_contours(img_no_bg, mask)
    centroid = find_centroid(mask)
    
    img_white_bg = transform_alpha_mask_to_white_background(img_with_contours)
    if centroid:
        save_contours(contours, centroid)
    
    plt.imshow(img_white_bg)
    plt.title("Imagem com Contornos")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
