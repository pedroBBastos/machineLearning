import scipy as scipy
import numpy as np


# Core — A point is a core point if it has more than MinPts points within eps. Border — This is a point that has
# at least one Core point at a distance n. Noise — This is a point that is neither a Core nor a Border. And it has
# less than m points within distance n from itself.

def DBSCAN(data, eps, minPts):
    pontos, dimensoes = data.shape
    matrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))
    visitados = np.zeros((pontos, 1))  # 0 ou 1
    cluster = -1 * np.ones((pontos, 1))  # -1 (no cluster)

    data = np.hstack((data, visitados))
    data = np.hstack((data, cluster))

    clusterNumber = 0
    for i in range(pontos):
        walkThroughPontosDaDensidade(i, matrixDeDistancia, data, eps, minPts, clusterNumber)
        clusterNumber += 1

    return data


def walkThroughPontosDaDensidade(i, matrixDeDistancia, data, eps, minPts, clusterNumber):
    if data[i][2] == 0:
        # ponto i visitado
        data[i][2] = 1

        # verificar se o ponto i possui minPts
        distanciasPontoAtual = matrixDeDistancia[i]
        pontosWithinEps = np.where(distanciasPontoAtual <= eps)[0]
        if pontosWithinEps.shape[0] >= minPts:
            data[pontosWithinEps, 3] = clusterNumber  # pontos dentro de eps pertencem ao cluster atual

            for p in pontosWithinEps:
                walkThroughPontosDaDensidade(p, matrixDeDistancia, data, eps, minPts, clusterNumber)
