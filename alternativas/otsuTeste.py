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

    

    return remove_fundo(mascara_pulmao)

# Teste do algoritmo da remoção do fundo
imagem_dcm = carregar_imagem("data/pulmao2/90.dcm")
imagem_hu = hu.converter_hu_para_cinza(imagem_dcm)
imagem_suavizada = cv2.GaussianBlur(imagem_hu, (5, 5), 0)
_, mascara_pulmao = cv2.threshold(imagem_suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Obter tanto a imagem com contornos quanto o dicionário de contornos válidos
imagem_contornos_somente, contornos_validos_dict = aplicar_otsu(imagem_hu)

# Plotar as imagens
plt.figure(figsize=(5, 5))
plt.imshow(imagem_hu, cmap='gray')
plt.axis('off')
plt.title("Imagem Original")

plt.figure(figsize=(5, 5))
plt.imshow(imagem_contornos_somente, cmap='gray')
plt.axis('off')
plt.title("Contorno Original")

# Criar uma imagem em branco para desenhar os contornos
imagem_contornos = np.zeros((imagem_hu.shape[0], imagem_hu.shape[1], 3), dtype=np.uint8)

# Converter os contornos do dicionário de volta para o formato NumPy
contornos_validos = [np.array(contorno, dtype=np.int32).reshape(-1, 1, 2) for contorno in contornos_validos_dict.values()]

# Desenhar os contornos válidos na imagem
cv2.drawContours(imagem_contornos, contornos_validos, -1, (0, 0, 255), 2)  # Vermelho, espessura 2

# Plotar a imagem com os contornos
plt.figure(figsize=(5, 5))
plt.imshow(imagem_contornos)
plt.axis('off')
plt.title("Contornos Válidos")

plt.show()