import numpy as np


def kMeans(data):
    # escolhendo um valor para k (nº de cluster)
    k = 3

    linhas = data.shape[0]
    colunaCluster = -1 * np.zeros((linhas, 1))
    data = np.hstack((data, colunaCluster))

    # escolhendo randomicamente k posições no espaço para serem os centroides
    centroides = np.random.randint(low=0, high=1000, size=(k, 2)) / 1000

    iterations = 0
    while True:

        for ponto in data:
            # obtendo as distancias para os centroides do ponto atual
            normsFromCentroides = np.linalg.norm(ponto[:2] - centroides, axis=1)
            # computando o centroide mais próximo
            nearestCentroideIndex = np.where(normsFromCentroides == np.amin(normsFromCentroides))[0][0]
            # indicando no ponto o indice do centroide ao qual ele pertence
            ponto[2] = nearestCentroideIndex

        # recalculando as coordenadas dos centroides
        newCentroides = np.zeros((k, 2))
        for i in range(0, centroides.shape[0]):
            pontosClusterAtual = data[data[:, 2] == i]
            pontosClusterAtual = pontosClusterAtual[:, :2]
            if pontosClusterAtual.size != 0:
                newCentroide = np.mean(pontosClusterAtual, axis=0)
                newCentroides[i] = newCentroide

        # verificando se os centroides pararam de se mover
        comparison = newCentroides == centroides
        centroides = newCentroides
        if comparison.all():
            break

        # com a normalização caiu bem a quantidade de interações
        iterations += 1

    print(iterations)
    return data
