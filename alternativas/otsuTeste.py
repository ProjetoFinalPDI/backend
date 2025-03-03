import cv2
import numpy as np
from segmentacao.remove_fundo import remove_fundo
from segmentacao.carregar import carregar_imagem
import segmentacao.hu as hu
import matplotlib.pyplot as plt

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

    # Encontrar contornos
    contornos, _ = cv2.findContours(mascara_pulmao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Converter de escala de cinza para BGR
    imagem_bgr = cv2.cvtColor(imagem_cinza, cv2.COLOR_GRAY2BGR)

    # Desenhar contornos na imagem original (em vermelho)
    cv2.drawContours(imagem_bgr, contornos, -1, (0, 0, 255), 1)

    # Criar uma imagem em preto e branco apenas com os contornos
    contornos_imagem = np.zeros_like(imagem_cinza)  # Criar uma imagem preta
    cv2.drawContours(contornos_imagem, contornos, -1, 255, 1)  # Desenhar contornos em branco

    return mascara_pulmao

# Teste do algoritmo da remoção do fundo
imagem_dcm = carregar_imagem("data/pulmao2/90.dcm")
imagem_hu = hu.converter_hu_para_cinza(imagem_dcm)
imagem_suavizada = cv2.GaussianBlur(imagem_hu, (5,5), 0)
_, mascara_pulmao = cv2.threshold(imagem_suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Obter tanto a imagem com contornos quanto a imagem somente com contornos
imagem_contornos_somente = aplicar_otsu(imagem_hu)

# Remover o fundo e preencher os contornos (usando a função otimizada)

imagem_sem_fundo = remove_fundo(imagem_contornos_somente, area_maxima=40000)

# Plotar as imagens
plt.figure(figsize=(5, 5))
plt.imshow(imagem_hu, cmap='gray')
plt.axis('off')
plt.title("Imagem Original")

plt.figure(figsize=(5, 5))
plt.imshow(imagem_contornos_somente, cmap='gray')
plt.axis('off')
plt.title("Contorno Original")


plt.figure(figsize=(5, 5))
plt.imshow(imagem_sem_fundo, cmap='gray')
plt.axis('off')
plt.title("Novo contorno sem fundo")

plt.show()
