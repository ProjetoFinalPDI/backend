import cv2
import numpy as np
from otsu import aplicar_otsu
from segmentacao.remove_fundo import remove_fundo

def dividir_imagem(imagem: np.ndarray, grid_size: tuple, overlap: int) -> list:
    """
    Divide a imagem em subimagens com sobreposição definida.
    
    Parâmetros:
        imagem (np.ndarray): Imagem de entrada em escala de cinza.
        grid_size (tuple): Número de divisões na vertical e horizontal.
        overlap (int): Quantidade de pixels de sobreposição entre as divisões.

    Retorna:
        list: Lista de tuplas contendo os limites e as subimagens extraídas.
    """
    altura, largura = imagem.shape
    sub_imagens = []
    
    passo_y = (altura + overlap * (grid_size[0] - 1)) // grid_size[0]
    passo_x = (largura + overlap * (grid_size[1] - 1)) // grid_size[1]
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y_inicio = max(0, i * passo_y - overlap)
            y_fim = min(altura, (i + 1) * passo_y + overlap)
            x_inicio = max(0, j * passo_x - overlap)
            x_fim = min(largura, (j + 1) * passo_x + overlap)
            
            sub_imagem = imagem[y_inicio:y_fim, x_inicio:x_fim]
            sub_imagens.append(((y_inicio, y_fim, x_inicio, x_fim), sub_imagem))
    
    return sub_imagens

def combinar_sub_imagens(imagem: np.ndarray, sub_imagens: list) -> np.ndarray:
    """
    Reconstrói a imagem combinando as subimagens processadas.
    
    Parâmetros:
        imagem (np.ndarray): Imagem original para referência de tamanho.
        sub_imagens (list): Lista de subimagens processadas.
    
    Retorna:
        np.ndarray: Imagem final combinada.
    """
    imagem_final = np.zeros_like(imagem)
    
    for (y_inicio, y_fim, x_inicio, x_fim), sub_imagem in sub_imagens:
        imagem_final[y_inicio:y_fim, x_inicio:x_fim] = sub_imagem
    
    return imagem_final

def segmentar_imagem(imagem: np.ndarray, grid_size: tuple = (2, 3), overlap: int = 20) -> np.ndarray:
    """
    Realiza a segmentação da imagem aplicando o método de Otsu em sub-regiões.
    
    Parâmetros:
        imagem (np.ndarray): Imagem a ser processada.
        grid_size (tuple): Tamanho da grade de divisão.
        overlap (int): Sobreposição entre as subimagens.
    
    Retorna:
        np.ndarray: Imagem segmentada final com remoção de fundo.
    """
    sub_imagens = dividir_imagem(imagem, grid_size, overlap)
    
    sub_imagens_processadas = [
        (limites, aplicar_otsu(sub_imagem)) for limites, sub_imagem in sub_imagens
    ]
    
    imagem_segmentada = combinar_sub_imagens(imagem, sub_imagens_processadas)
    
    return remove_fundo(imagem_segmentada)
