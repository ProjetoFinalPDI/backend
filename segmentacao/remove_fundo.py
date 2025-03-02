import cv2
import numpy as np

def remove_fundo(imagem: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """
    Remove o fundo da imagem e mantém apenas os contornos da região segmentada.

    Parâmetros:
        imagem (np.ndarray): Imagem original (em escala de cinza ou BGR).
        mascara (np.ndarray): Máscara binária da segmentação.

    Retorna:
        np.ndarray: Imagem com os contornos destacados.
    """
    # Encontrar contornos na máscara binária
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar uma imagem preta do mesmo tamanho
    imagem_contornos = np.zeros_like(imagem)

    # Se a imagem for em escala de cinza, converter para BGR
    if len(imagem.shape) == 2:
        imagem_contornos = cv2.cvtColor(imagem_contornos, cv2.COLOR_GRAY2BGR)

    # Desenhar apenas os contornos na imagem preta
    cv2.drawContours(imagem_contornos, contornos, -1, (0, 255, 0), 1)  # Contornos em verde

    return imagem_contornos
