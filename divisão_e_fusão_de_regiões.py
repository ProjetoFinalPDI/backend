import cv2
import numpy as np
from segmentacao.carregar import carregar_imagem
import alternativas.hu_para_cinza as hu
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy

from datetime import datetime
inicio = datetime.now()

def criterio_homogeneidade(region: np.ndarray, limite_var: float) -> bool:
    """
    Verifica se uma região é homogênea com base na variância.
    """
    return np.var(region) < limite_var

def criterio_media(region: np.ndarray, referencia: float, limite_media: float) -> bool:
    """Verifica se a média da região está próxima da referência."""
    return abs(np.mean(region) - referencia) < limite_media

def aplicar_divisao_e_fusao(imagem: np.ndarray, limite_var: float, limite_media: float, referencia_media: float) -> np.ndarray:
    """
    Aplica o algoritmo de Divisão e Fusão de Regiões para segmentação de pulmões.

    Parâmetros:
        imagem (np.ndarray): Imagem de entrada em escala de cinza.
        limite_var (float): Limite de variância para determinar a homogeneidade da região.

    Retorna:
        np.ndarray: Imagem segmentada com os contornos dos pulmões destacados.

    Resumo da teoria:
        Técnica consiste em fazer divisões na imagem principal e agrupar os blocos formados dessas divisões 
        baseado em um critério de junção (verifica se a região definida pelos blocos consdierados pode ser dita homogênea pelo critério). 
        Ex.: 
        1 - critério de variância: Dado um limite de variância definido, checa se a variância dos valores do pixels está acima 
                                   ou abaixo desse limite, se abaixo, cumpre o criteŕio, se acima, não o cumpre.
        2 - critério de média: Dado um valor de limite de média e uma referência, calcula-se a 
    """
    altura, largura = imagem.shape
    tamanho_min = 1  # Tamanho mínimo da região
    segmentos = np.zeros_like(imagem, dtype=np.uint8)

    # Aplicação de filtro Gaussiano para tentar melhorar a performance
    #imagem_suavizada = cv2.GaussianBlur(imagem, (3, 3), 0)
    
    def dividir(x, y, tamanho):
        """ Divide recursivamente a região se não for homogênea. """
        if tamanho < tamanho_min:
            return
        subregiao = imagem[y:y+tamanho, x:x+tamanho]
        if criterio_homogeneidade(subregiao, limite_var) and criterio_media(subregiao, referencia_media, limite_media):
            segmentos[y:y+tamanho, x:x+tamanho] = 255
        else:
            metade = tamanho // 2
            dividir(x, y, metade)
            dividir(x + metade, y, metade)
            dividir(x, y + metade, metade)
            dividir(x + metade, y + metade, metade)
    
    dividir(0, 0, min(altura, largura))

    '''mascara = np.zeros_like(imagem_suavizada, dtype=np.uint8)
    for (x, y, w, h) in segmentos:
        mascara[y:y+h, x:x+w] = 255'''

    kernel = np.ones((5, 5), np.uint8)
    segmentos = cv2.morphologyEx(segmentos, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fecha buracos
    segmentos = cv2.morphologyEx(segmentos, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove ruídos pequenos
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(segmentos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # segmentos
    
    # Converter de escala de cinza para BGR
    imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    
    # Desenhar contornos em azul
    cv2.drawContours(imagem_bgr, contornos, -1, (0, 0, 255), 1)
    
    return imagem_bgr

imagem_dcm = carregar_imagem("data/pulmao2/155.dcm")
imagem_hu = hu.converter_hu_para_cinza(imagem_dcm)
imagem_div_fus = aplicar_divisao_e_fusao(imagem=imagem_hu, limite_var=40, limite_media=40, referencia_media=5)

fim = datetime.now()
tempo_execucao = fim - inicio
print(f"Tempo de execução: {tempo_execucao}")

plt.figure(figsize=(5, 5))
plt.imshow(imagem_div_fus, cmap='gray')
plt.axis('off')
plt.show()
