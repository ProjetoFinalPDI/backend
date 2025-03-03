import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentacao.hu import converter_hu_para_cinza
from segmentacao.carregar import carregar_imagem
from segmentacao.remove_fundo import remove_fundo



def aplicar_watershed(imagem_cinza: np.ndarray) -> np.ndarray:
    """
    Aplica o algoritmo Watershed para segmentação dos pulmões em toda a imagem.

    Parâmetros:
        imagem_cinza (np.ndarray): Imagem de entrada em escala de cinza (512x512).

    Retorna:
        np.ndarray: Máscara binária com os contornos dos pulmões destacados em branco (255) sobre fundo preto (0).
    """
    # Criar uma máscara binária para armazenar os contornos
    mascara_contornos = np.zeros_like(imagem_cinza, dtype=np.uint8)

    # Aplicar CLAHE para melhorar o contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagem_equalizada = clahe.apply(imagem_cinza)

    # Suavizar a imagem com um filtro Gaussiano
    imagem_suavizada = cv2.GaussianBlur(imagem_equalizada, (5, 5), 0)

    # Aplicar threshold para destacar os pulmões
    _, mascara_pulmao = cv2.threshold(imagem_suavizada, 50, 255, cv2.THRESH_BINARY_INV)

    # Aplicar operações morfológicas para refinar a máscara
    kernel = np.ones((5, 5), np.uint8)
    mascara_pulmao = cv2.morphologyEx(mascara_pulmao, cv2.MORPH_CLOSE, kernel, iterations=3)
    mascara_pulmao = cv2.morphologyEx(mascara_pulmao, cv2.MORPH_OPEN, kernel, iterations=2)

    # Criar marcadores para Watershed
    # sure_bg = cv2.dilate(mascara_pulmao, kernel, iterations=3)
    # dist_transform = cv2.distanceTransform(mascara_pulmao, cv2.DIST_L2, 5)
    # _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    # # Criar marcadores
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    # _, marcadores = cv2.connectedComponents(sure_fg)
    # marcadores = marcadores + 1
    # marcadores[unknown == 255] = 0

    # # Aplicar Watershed na imagem inteira
    # imagem_colorida = cv2.cvtColor(imagem_cinza, cv2.COLOR_GRAY2BGR)
    # marcadores = cv2.watershed(imagem_colorida, marcadores)

    # # Criar uma máscara com os contornos do Watershed (branco = 255, fundo preto = 0)
    # mascara_contornos[marcadores == -1] = 255  # Contornos em branco

    return remove_fundo(mascara_pulmao)


# Carregar imagem DICOM e converter para escala de cinza
imagem_hu = carregar_imagem('data/pulmao2/157.dcm')
imagem_cinza = converter_hu_para_cinza(imagem_hu)

# Aplicar Watershed para obter apenas os contornos
imagem_contornos = aplicar_watershed(imagem_cinza)


# Plotar as imagens
plt.figure(figsize=(5, 5))
plt.imshow(imagem_hu, cmap='gray')
plt.axis('off')
plt.title("Imagem Original")

plt.figure(figsize=(5, 5))
plt.imshow(imagem_contornos, cmap='gray')
plt.axis('off')
plt.title("Contorno")

plt.show()