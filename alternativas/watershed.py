import cv2
import numpy as np

def aplicar_watershed(imagem_cinza: np.ndarray) -> np.ndarray:
    """
    Aplica o algoritmo Watershed para segmentação dos pulmões em imagens com
    a técnica Watershed, implementada na biblioteca cv2.

    Parâmetros:
        imagem_cinza (np.ndarray): Pixels da imagem de entrada em escala de cinza.

    Retorna:
        np.ndarray: Pixels da imagem original com os contornos dos pulmões
        destacados em vermelho.
    """
    # Aplicar equalização de histograma para melhorar o contraste
    equalizacao = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagem_equalizada = equalizacao.apply(imagem_cinza)

    # Aplicar um filtro Gaussiano para suavizar a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_equalizada, (5,5), 0)

    # Aplicar threshold para destacar os pulmões, intensidades com valor menor do que 80
    # tornam-se brancas
    _, mascara_pulmao = cv2.threshold(imagem_suavizada, 90, 255, cv2.THRESH_BINARY_INV)

    # Aplicar fechamento morfológico para preencher pequenas falhas
    kernel = np.ones((5,5), np.uint8)
    mascara_pulmao = cv2.morphologyEx(mascara_pulmao, cv2.MORPH_CLOSE, kernel,
                                      iterations=3)
    mascara_pulmao = cv2.morphologyEx(mascara_pulmao, cv2.MORPH_OPEN, kernel,
                                      iterations=2)

    # Criar os marcadores para Watershed
    certamente_fundo = cv2.dilate(mascara_pulmao, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(mascara_pulmao, cv2.DIST_L2, 5)
    _, certamente_primeiro_plano = cv2.threshold(dist_transform,
                                                 0.3 * dist_transform.max(),
                                                 255, 0)

    # Converter certamente_primeiro_plano para uint8
    certamente_primeiro_plano = np.uint8(certamente_primeiro_plano)
    incerto = cv2.subtract(certamente_fundo, certamente_primeiro_plano)

    # Criar marcadores
    _, marcadores = cv2.connectedComponents(certamente_primeiro_plano)
    marcadores = marcadores + 1
    marcadores[incerto == 255] = 0

    # Converter de escala de cinza para BGR
    imagem_bgr = cv2.cvtColor(imagem_cinza, cv2.COLOR_GRAY2BGR)

    # Aplicar Watershed
    marcadores = cv2.watershed(imagem_bgr, marcadores)

    # Destacar contornos dos pulmões em vermelho
    imagem_bgr[marcadores == -1] = [0, 0, 255]

    return imagem_bgr
