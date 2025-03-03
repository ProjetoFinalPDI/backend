import cv2
import numpy as np
from segmentacao.carregar import carregar_imagem
import alternativas.hu_para_cinza as hu
import matplotlib.pyplot as plt

def criterio_homogeneidade(region: np.ndarray, limite_var: float) -> bool:
    """ Verifica se uma região é homogênea com base na variância. """
    return np.var(region) < limite_var

def criterio_media(region: np.ndarray, referencia: float, limite_media: float) -> bool:
    """ Verifica se a média da região está próxima da referência. """
    return abs(np.mean(region) - referencia) < limite_media

def aplicar_divisao_e_fusao(imagem: np.ndarray, limite_var: int, limite_media: int, referencia_media: int) -> np.ndarray:
    """
    Aplica o algoritmo de Divisão e Fusão de Regiões para segmentação de pulmões.

    Parâmetros:
        imagem (np.ndarray): Imagem de entrada em escala de cinza.
        limite_var (int): Limite de variância.
        limite_media (int): Limite de média. 
        referencia_media (int): Referência de média para cálculo da diferença entre médias.

    Retorna:
        np.ndarray: Imagem segmentada com os contornos dos pulmões destacados.

    Resumo da teoria:
        Técnica consiste em fazer divisões na imagem principal e agrupar os blocos formados dessas divisões 
        baseado em um critério de junção, caso o critério falhe, as regiões são novamente divididas até que se chegue
        a uma região de tamanho unitário. 
        
        Critérios de homogeneidade escolhidos: 
        1 - Variância de intensidades: checa se a variância da região considerada excede o limite;
        2 - Média de intensidades: checa se a diferença entre a média da região e um valor referência excede um limite;
                                   
        Caso ambos os critérios sejam satisfeitos, a região não é dividida; se ao menos um não satisfazer, ela é novamente
        dividida e os critérios são avaliados para cada nova região criada.
    """
    altura, largura = imagem.shape
    tamanho_min = 1  # Tamanho mínimo da região
    segmentos = np.zeros_like(imagem, dtype=np.uint8)

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
    
    # Aplicar divisão recursiva
    dividir(0, 0, min(altura, largura))

    # Operações morfológicas vetorizadas
    kernel = np.ones((5, 5), np.uint8)
    segmentos = cv2.morphologyEx(segmentos, cv2.MORPH_CLOSE, kernel, iterations=2)  # Fecha buracos
    segmentos = cv2.morphologyEx(segmentos, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove ruídos pequenos

    # Encontrar contornos
    contornos, _ = cv2.findContours(segmentos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Converter de escala de cinza para BGR
    imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    # Desenhar contornos em azul
    cv2.drawContours(imagem_bgr, contornos, -1, (0, 0, 255), 1)

    return imagem_bgr

# Carregar e processar a imagem
imagem_dcm = carregar_imagem("data/pulmao2/155.dcm")
imagem_hu = hu.converter_hu_para_cinza(imagem_dcm)
imagem_div_fus = aplicar_divisao_e_fusao(imagem=imagem_hu, limite_var=40, limite_media=40, referencia_media=5)

# Exibir a imagem resultante
plt.figure(figsize=(5, 5))
plt.imshow(imagem_div_fus, cmap='gray')
plt.axis('off')
plt.show()