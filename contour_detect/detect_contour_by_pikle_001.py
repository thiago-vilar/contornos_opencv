import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from rembg import remove
import stag

def load_contour_from_file(file_path):
    """Carrega um contorno salvo em formato .pkl"""
    with open(file_path, 'rb') as file:
        contour = pickle.load(file)
    return contour

def compare_contours(contour1, contour2):
    """Compara dois contornos usando cv2.matchShapes"""
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)

def detect_and_label_stags(image_path, library_hd=17, error_correction=None):
    """Detecta as stags presentes na imagem"""
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem.")
        return None, None, None

    config = {'libraryHD': library_hd}
    if error_correction is not None:
        config['errorCorrection'] = error_correction

    corners, ids, _ = stag.detectMarkers(image, **config)

    if ids is None:
        print("Nenhum marcador foi encontrado.")
        return None, None, image

    return corners, ids, image

def display_markers(image, corners, ids):
    """Mostra os ids das stags e calcula as áreas de varredura"""
    scan_areas = {}
    for corner, id_ in zip(corners, ids.flatten()):
        corner = corner.reshape(-1, 2).astype(int)
        centroid_x = int(np.mean(corner[:, 0]))
        centroid_y = int(np.mean(corner[:, 1]))

        # Desenha o contorno e o ID do marcador
        cv2.polylines(image, [corner], True, (0, 255, 0), 2)
        cv2.putText(image, f'ID: {id_}', (centroid_x, centroid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Calcular dimensões da área de varredura (ScanArea)
        width = np.max(corner[:, 0]) - np.min(corner[:, 0])
        height = np.max(corner[:, 1]) - np.min(corner[:, 1])
        x_min = centroid_x - width // 2
        x_max = centroid_x + width // 2
        y_min = centroid_y - height // 2
        y_max = centroid_y + height // 2

        # Armazenar as coordenadas da área de varredura
        scan_areas[id_] = (x_min, x_max, y_min, y_max)

    # Exibe a imagem com os IDs das stags
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Selecione o ID da área de varredura")
    plt.axis('off')
    plt.show()

    return scan_areas

def crop_scan_area(image, scan_areas, selected_id):
    """Corta a imagem na área de varredura selecionada pelo usuário com base no ID"""
    if selected_id not in scan_areas:
        print(f"ID {selected_id} não encontrado.")
        return None
    x_min, x_max, y_min, y_max = scan_areas[selected_id]
    return image[y_min:y_max, x_min:x_max]

def remove_background(image_np_array):
    """Remove o fundo de uma imagem numpy array"""
    is_success, buffer = cv2.imencode(".jpg", image_np_array)
    if not is_success:
        raise ValueError("Falha ao codificar a imagem para remoção de fundo.")

    output_image = remove(buffer.tobytes())
    img = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Falha ao decodificar a imagem processada.")
    return img[:, :, :3] 

def create_mask(img):
    """Cria uma máscara binária para a imagem com base em um intervalo de cores"""
    lower_bound = np.array([30, 30, 30])
    upper_bound = np.array([250, 250, 250])
    return cv2.inRange(img, lower_bound, upper_bound)

def extract_contours(img, mask):
    """Extrai os contornos da máscara"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    # Solicitar o caminho do contorno de referência salvo em .pkl
    reference_contour_path = input("Digite o caminho para o contorno de referência (.pkl): ")
    reference_contour = load_contour_from_file(reference_contour_path)

    # Solicitar o caminho da imagem com as stags
    filepath = input("Digite o caminho da imagem: ")
    corners, ids, image = detect_and_label_stags(filepath)

    if corners is not None:
        # Mostrar as áreas de varredura
        scan_areas = display_markers(image, corners, ids)

        # Solicitar ID da área de varredura a ser processada
        selected_id = int(input("Digite o ID da área de varredura a ser processada: "))
        cropped_image = crop_scan_area(image, scan_areas, selected_id)

        if cropped_image is not None:
            # Remover o fundo
            bg_removed_image = remove_background(cropped_image)

            # Criar máscara
            mask = create_mask(bg_removed_image)

            # Extrair contornos
            contours = extract_contours(bg_removed_image, mask)

            if contours:
                # Comparar o maior contorno extraído com o contorno de referência
                largest_contour = max(contours, key=cv2.contourArea)
                similarity = compare_contours(reference_contour, largest_contour)

                # Exibir resultado da comparação
                print(f"Similaridade: {similarity:.4f}")
                if similarity < 0.1:
                    print("Os contornos são semelhantes.")
                else:
                    print("Os contornos são diferentes.")

    else:
        print("Nenhum marcador foi detectado na imagem.")

if __name__ == "__main__":
    main()
