import cv2
import numpy as np

def remove_fundo(mascara: np.ndarray, area_maxima: int = 40000) -> np.ndarray:
    """
    Mantém apenas contornos fechados cujas áreas não excedem a área máxima especificada e que não tocam a borda da imagem.

    Parâmetros:
        mascara (np.ndarray): Máscara binária com os contornos.
        area_maxima (int): Área máxima permitida para os contornos (default: 40000).

    Retorna:
        np.ndarray: Imagem com os contornos preenchidos em branco e o fundo preto.
    """
    # Encontrar contornos na máscara
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar uma máscara para preencher os contornos
    mascara_preenchida = np.zeros_like(mascara)

    # Obter as dimensões da imagem
    altura, largura = mascara.shape

    # Filtrar e preencher apenas os contornos fechados que não tocam a borda e têm áreas menores ou iguais à área máxima
    for contorno in contornos:
        # Verificar se o contorno é fechado
        if cv2.arcLength(contorno, True) > 0:  # Verifica se o contorno tem comprimento positivo
            # Verificar se o contorno toca a borda da imagem
            toca_borda = False
            for ponto in contorno:
                x, y = ponto[0]
                if x == 0 or x == largura - 1 or y == 0 or y == altura - 1:
                    toca_borda = True
                    break

            # Se o contorno não tocar a borda e tiver área menor ou igual à área máxima, preencher
            if not toca_borda:
                area = cv2.contourArea(contorno)
                if area <= area_maxima:
                    cv2.drawContours(mascara_preenchida, [contorno], -1, 255, -1)  # Preencher o contorno

    return mascara_preenchida