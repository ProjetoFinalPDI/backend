import cv2
import numpy as np

def aplicar_otsu(imagem_cinza: np.ndarray) -> np.ndarray:
    """
    Aplica o algoritmo de Otsu para segmentação dos pulmões em imagens através da biblioteca cv2.

    Parâmetros:
        imagem_cinza (np.ndarray): Pixels da imagem de entrada em escala de cinza.

    Retorna:
        np.ndarray: Pixels da imagem original com os contornos dos pulmões
        destacados em vermelho.
    """

    # Aplica um filtro Gaussiano para suavizar a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5,5), 0)

    # Aplica threshold de Otsu para segmentação
    _, mascara_pulmao = cv2.threshold(imagem_suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontra contornos
    contornos, _ = cv2.findContours(mascara_pulmao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Converte de escala de cinza para BGR
    imagem_bgr = cv2.cvtColor(imagem_cinza, cv2.COLOR_GRAY2BGR)

    # Desenha contornos em azul
    cv2.drawContours(imagem_bgr, contornos, -1, (0, 0, 255), 1)

    return imagem_bgr