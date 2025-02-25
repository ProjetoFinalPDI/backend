import numpy as np
from matplotlib.path import Path
from energia import energia_interna_adaptativa
from carregar import carregar_imagem
from classificacao import calcula_ocorrencias_classes, probabilidade_classes


def adicionar_pontos(
    curva: np.ndarray, imagem: np.ndarray, d: float, w_adapt=0.1, w_cont=0.6
) -> np.ndarray:
    """
    Adiciona pontos à curva minimizando a energia e garantindo que pertencem ao pulmão.

    Args:
        curva (np.ndarray): Pontos da curva inicial.
        imagem (np.ndarray): Imagem DICOM carregada em Unidades Hounsfield.
        d (float): Distância mínima entre pontos.
        w_adapt (float): Peso da energia adaptativa.
        w_cont (float): Peso da energia de continuidade.

    Returns:
        np.ndarray: Curva refinada com pontos adicionados.
    """
    nova_curva = [curva[0]]
    poligono = Path(curva)  # Criar polígono da curva para verificação de inclusão

    # Calcular probabilidades das classes pulmonares
    ocorrencias = calcula_ocorrencias_classes(imagem)
    probabilidades = probabilidade_classes(ocorrencias)

    for i in range(len(curva) - 1):
        p1, p2 = curva[i], curva[i + 1]
        dist = np.linalg.norm(p2 - p1)

        if dist > d:
            num_pontos = int(dist // d)
            melhor_pontos = []

            for j in range(1, num_pontos + 1):
                candidato = p1 + (p2 - p1) * (j / (num_pontos + 1))

                # Verificar se o ponto está dentro da curva
                if poligono.contains_point(candidato):
                    x, y = int(candidato[0]), int(candidato[1])

                    # Verificar se o ponto pertence ao pulmão (hiperaerado e normalmente aerado)
                    if -1000 <= imagem[y, x] <= -500:
                        energia = energia_interna_adaptativa(curva, i, w_adapt, w_cont)
                        prob_pulmao = (
                            probabilidades[0, y, x] + probabilidades[1, y, x]
                        )  # Soma das classes pulmonares

                        # Só adiciona o ponto se a probabilidade de pulmão for alta
                        if prob_pulmao > 0.5:
                            melhor_pontos.append((energia, candidato))

            # Ordenar os pontos pela menor energia
            melhor_pontos.sort(key=lambda x: x[0])
            nova_curva.extend([p[1] for p in melhor_pontos])

        nova_curva.append(p2)

    return np.array(nova_curva)
