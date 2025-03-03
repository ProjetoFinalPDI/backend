import cv2
import numpy as np

def remove_fundo(mascara: np.ndarray, area_minima: int = 3000, area_maxima: int = 40000) -> tuple:
    """
    Mantém apenas contornos fechados cujas áreas estão dentro do intervalo especificado e que não tocam a borda da imagem.

    Parâmetros:
        mascara (np.ndarray): Máscara binária com os contornos.
        area_minima (int): Área mínima permitida para os contornos (default: 3000).
        area_maxima (int): Área máxima permitida para os contornos (default: 40000).

    Retorna:
        tuple:
            - np.ndarray: Imagem com os novos contornos preenchidos em vermelho.
            - list: Lista de arrays, onde cada array representa um contorno válido.
    """
    # Encontrar contornos na máscara
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar uma imagem para os contornos vermelhos
    pulmao_contornado = np.zeros((mascara.shape[0], mascara.shape[1], 3), dtype=np.uint8)  # Imagem em preto

    # Lista para armazenar os contornos válidos
    contornos_validos = []

    # Obter as dimensões da imagem
    altura, largura = mascara.shape

    # Filtrar e desenhar apenas os contornos fechados que não tocam a borda e têm áreas dentro do intervalo especificado
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

            # Se o contorno não tocar a borda e tiver área dentro do intervalo, desenhar em vermelho e adicionar à lista
            if not toca_borda:
                area = cv2.contourArea(contorno)
                if area_minima <= area <= area_maxima:
                    # Desenhar o contorno válido em vermelho na imagem de contornos vermelhos
                    cv2.drawContours(pulmao_contornado, [contorno], -1, (0, 0, 255), 2)  # Vermelho
                    # Adicionar o contorno à lista de contornos válidos
                    contornos_validos.append(contorno)

    return pulmao_contornado, contornos_validos