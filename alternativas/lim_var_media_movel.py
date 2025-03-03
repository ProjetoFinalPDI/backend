import cv2
import numpy as np

def limiarizacao_media_movel(imagem, window_size=31, k=0.2):
    limiarizada = np.zeros_like(imagem)
    
    # calculando a média móvel usando uma janela deslizante
    half_window = window_size // 2
    for i in range(half_window, imagem.shape[0] - half_window):
        for j in range(half_window, imagem.shape[1] - half_window):
            # extraindo a janela local
            janela = imagem[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            
            # média local
            media = np.mean(janela)
            
            # aplicando o limiar: se o valor do pixel for maior que o limiar, o pixel é 1 (objeto), caso contrário, 0 (fundo)
            limiarizada[i, j] = 255 if imagem[i, j] > media + k * media else 0
    
    return limiarizada