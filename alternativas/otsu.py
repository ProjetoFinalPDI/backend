import cv2
import numpy as np
from segmentacao.remove_fundo import remove_fundo
from segmentacao.carregar import carregar_imagem
import segmentacao.hu as hu

def aplicar_otsu(imagem_cinza: np.ndarray) -> tuple:
    """
    Aplica o algoritmo de Otsu para segmentação dos pulmões em imagens.

    Parâmetros:
        imagem_cinza (np.ndarray): Pixels da imagem de entrada em escala de cinza.

    Retorna:
        tuple: 
            - Imagem original com os contornos dos pulmões destacados em vermelho.
            - Imagem com apenas os contornos dos pulmões em branco sobre fundo preto.
    """
    # Aplicar um filtro Gaussiano para suavizar a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5,5), 0)

    # Aplicar threshold de Otsu para segmentação
    _, mascara_pulmao = cv2.threshold(imagem_suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    return remove_fundo(mascara_pulmao)
