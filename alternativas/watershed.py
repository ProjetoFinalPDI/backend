from segmentacao.carregar import carregar_imagem

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

imagem_original_em_hu = carregar_imagem("data/pulmao2/1.dcm")

print(imagem_original_em_hu)