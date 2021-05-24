import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

import kMeans as km
import elbow as em


def executeKMeans(dataTraining, dataTest):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(dataTraining)

    pontosTreino, dimensoes = data.shape

    centroidesPorK = []
    clusteredPorK = []
    clusteredNumbersPorK = []
    k_clusters = [1, 2, 3, 4, 5]
    elbow_values_plot = []

    for k in k_clusters:
        clustered, centroides = km.kMeans(data, k, dimensoes)
        clusterNumbers = np.unique(clustered[:, dimensoes])

        centroidesPorK.append(centroides)
        clusteredPorK.append(clustered)
        clusteredNumbersPorK.append(clusterNumbers)

        # for ci in clusterNumbers:
        #     ci = clustered[clustered[:, 2] == ci]
        #     ci = ci[:, :2]
        #     cix = ci[:, 0]
        #     ciy = ci[:, 1]
        #     plt.plot(cix, ciy, color=np.random.random(3), marker='x', linestyle='')
        # plt.show()

        value = em.elbow_value(clustered, centroides, dimensoes)
        elbow_values_plot.append(value)

    print(clusteredPorK)

    # em.plot_elbow_graphic(elbow_values_plot, k_clusters, 1)

    ################################################################
    # melhor k = 4
    # verificando para qual cluster pertencem os dados de teste
    ################################################################

    # dataTest = min_max_scaler.fit_transform(dataTest)
    #
    # linhas = dataTest.shape[0]
    # colunaCluster = -1 * np.zeros((linhas, 1))
    # dataTest = np.hstack((dataTest, colunaCluster))
    #
    # centroidesTeste = centroidesPorK[0]
    # clusteredTeste = clusteredPorK[0]
    # clusterNumbersTeste = clusteredNumbersPorK[0]
    #
    # for ponto in dataTest:
    #     # obtendo as distancias para os centroides do ponto atual
    #     normsFromCentroides = np.linalg.norm(ponto[:dimensoes] - centroidesTeste, axis=1)
    #     # computando o centroide mais próximo
    #     nearestCentroideIndex = np.where(normsFromCentroides == np.amin(normsFromCentroides))[0][0]
    #     # indicando no ponto o indice do centroide ao qual ele pertence
    #     ponto[dimensoes] = nearestCentroideIndex
    #
    # # np.append(clusteredTeste, dataTest[:, :2])
    #
    # for ci in clusterNumbersTeste:
    #     cor = np.random.random(3)
    #
    #     pontosCi = clusteredTeste[clusteredTeste[:, 2] == ci]
    #     pontosCi = pontosCi[:, :2]
    #     pontosCix = pontosCi[:, 0]
    #     pontosCiy = pontosCi[:, 1]
    #     plt.plot(pontosCix, pontosCiy, color=cor, marker='x', linestyle='')
    # plt.show()


def kMeans(data, k, dimensoes):

    maxIterations = k*100
    iterationsCount = 0

    linhas = data.shape[0]
    colunaCluster = -1 * np.zeros((linhas, 1))
    data = np.hstack((data, colunaCluster))

    # escolhendo randomicamente k posições no espaço para serem os centroides
    centroides = np.random.randint(low=0, high=1000, size=(k, dimensoes)) / 1000

    while True:

        for ponto in data:
            # obtendo as distancias para os centroides do ponto atual
            normsFromCentroides = np.linalg.norm(ponto[:dimensoes] - centroides, axis=1)
            # computando o centroide mais próximo
            nearestCentroideIndex = np.where(normsFromCentroides == np.amin(normsFromCentroides))[0][0]
            # indicando no ponto o indice do centroide ao qual ele pertence
            ponto[2] = nearestCentroideIndex

        # recalculando as coordenadas dos centroides
        newCentroides = np.zeros((k, dimensoes))
        for i in range(0, centroides.shape[0]):
            pontosClusterAtual = data[data[:, dimensoes] == i]
            pontosClusterAtual = pontosClusterAtual[:, :dimensoes]
            if pontosClusterAtual.size != 0:
                newCentroide = np.mean(pontosClusterAtual, axis=0)
                newCentroides[i] = newCentroide

        # verificando se os centroides pararam de se mover
        comparison = newCentroides == centroides
        centroides = newCentroides
        if comparison.all() or iterationsCount == maxIterations:
            break

        iterationsCount += 1

    return data, newCentroides
