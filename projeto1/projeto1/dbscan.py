import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt

from sklearn import preprocessing
import sklearn.metrics as sklMetrics

import elbow as em
import scipy.spatial


def executeDBSCAN(dataTraining, dataTest, eps_values, minPts):
    pontosTreino, _ = dataTraining.shape

    min_max_scaler = preprocessing.MinMaxScaler()
    dataTraining = min_max_scaler.fit_transform(dataTraining)

    matrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(dataTraining))

    # knee gráfico
    thirdMinDists = np.sort(np.partition(matrixDeDistancia, 4)[:, 4])
    em.plot_elbow_graphic(thirdMinDists, range(0, len(thirdMinDists)), 2)

    clusteredPorEps = []
    corePointsPorEps = []
    clusterNumbersPorEps = []

    for eps in eps_values:
        clustered, corePointsList = DBSCAN(dataTraining, matrixDeDistancia, eps, minPts)
        clusterNumbers = np.unique(clustered[:, 3])

        clusteredPorEps.append(clustered)
        corePointsPorEps.append(corePointsList)
        clusterNumbersPorEps.append(clusterNumbers)

        # silhouette_score como métrica para avaliar
        # silhouette dá erro se for executado com apenas um cluster
        if np.unique(clustered[:, 3]).shape[0] != 1:
            silhouette_score = sklMetrics.silhouette_score(dataTraining, clustered[:, 3])

        # for ci in clusterNumbers:
        #     ci = clustered[clustered[:, 3] == ci]
        #     ci = ci[:, :2]
        #     cix = ci[:, 0]
        #     ciy = ci[:, 1]
        #     plt.plot(cix, ciy, color=np.random.random(3), marker='x', linestyle='')
        # plt.show()

    ###########################################################
    # Test -> melhor EPS = 0.0864
    ###########################################################

    # clusteredPorEps = []
    # corePointsPorEps = []
    # clusterNumbersPorEps = []

    dataTest = min_max_scaler.fit_transform(dataTest)
    pontosTeste, _ = dataTest.shape

    chosenClustered = clusteredPorEps[0]
    chosenClustered = np.delete(chosenClustered, 2, 1)
    initCluster = -1 * np.ones((pontosTeste, 1))  # -1 (no cluster)
    dataTest = np.hstack((dataTest, initCluster))

    newMatrix = np.concatenate((chosenClustered[:, :2], dataTest[:, :2]))
    novaMatrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(newMatrix))
    novaMatrixDeDistancia = novaMatrixDeDistancia[pontosTreino:, :pontosTreino]
    print(novaMatrixDeDistancia)

    indicesPontosMaisProximos = np.argmin(novaMatrixDeDistancia, axis=1)
    mask = np.zeros((pontosTeste, pontosTreino), dtype=bool)
    mask[np.arange(len(mask)), indicesPontosMaisProximos] = True
    menoresDistancias = novaMatrixDeDistancia[mask]

    # atribuindo número de cluster aos novos pontos
    toAssignCluster = np.where(menoresDistancias <= eps_values[0])[0]
    dataTest[toAssignCluster, 2] = chosenClustered[indicesPontosMaisProximos[toAssignCluster], 2]

    clusterNumbersTeste = clusterNumbersPorEps[6]
    for ci in clusterNumbersTeste:
        cor = np.random.random(3)

        pontosCi = chosenClustered[chosenClustered[:, 2] == ci]
        pontosCi = pontosCi[:, :2]
        pontosCix = pontosCi[:, 0]
        pontosCiy = pontosCi[:, 1]
        plt.plot(pontosCix, pontosCiy, color=cor, marker='x', linestyle='')

        ciTeste = dataTest[dataTest[:, 2] == ci]
        ciTeste = ciTeste[:, :2]
        ciTestex = ciTeste[:, 0]
        ciTestey = ciTeste[:, 1]
        plt.plot(ciTestex, ciTestey, color=cor, marker='s', linestyle='')
    plt.show()


# Core — A point is a core point if it has more than MinPts points within eps. Border — This is a point that has
# at least one Core point at a distance n. Noise — This is a point that is neither a Core nor a Border. And it has
# less than m points within distance n from itself.

def DBSCAN(data, matrixDeDistancia, eps, minPts):
    pontos, dimensoes = data.shape
    visitados = np.zeros((pontos, 1))  # 0 ou 1
    cluster = -1 * np.ones((pontos, 1))  # -1 (no cluster)

    data = np.hstack((data, visitados))
    data = np.hstack((data, cluster))

    clusterNumber = 0
    corePointsList = []

    for i in range(pontos):
        walkThroughPontosDaDensidade(i, matrixDeDistancia, data,
                                     eps, minPts, clusterNumber,
                                     dimensoes, corePointsList)
        clusterNumber += 1

    return data, np.array(corePointsList)


def walkThroughPontosDaDensidade(i, matrixDeDistancia,
                                 data, eps, minPts,
                                 clusterNumber, dimensoes, corePointsList):
    if data[i][dimensoes] == 0:
        # ponto i visitado
        data[i][dimensoes] = 1

        # verificar se o ponto i possui minPts
        distanciasPontoAtual = matrixDeDistancia[i]
        pontosWithinEps = np.where(distanciasPontoAtual <= eps)[0]
        if pontosWithinEps.shape[0] >= minPts:
            corePointsList.append([i, clusterNumber])
            data[pontosWithinEps, dimensoes + 1] = clusterNumber  # pontos dentro de eps pertencem ao cluster atual

            for p in pontosWithinEps:
                walkThroughPontosDaDensidade(p, matrixDeDistancia, data,
                                             eps, minPts, clusterNumber,
                                             dimensoes, corePointsList)

# def executeDBSCAN3D(data):
#     min_max_scaler = preprocessing.MinMaxScaler()
#     data = min_max_scaler.fit_transform(data)
#
#     matrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data))
#
#     clustered = dbscan.DBSCAN(data, matrixDeDistancia, 0.13, 11)
#     clusterNumbers = np.unique(clustered[:, 4])
#
#     ax = plt.axes(projection='3d')
#     for ci in clusterNumbers:
#         ci = clustered[clustered[:, 4] == ci]
#         ci = ci[:, :3]
#         cix = ci[:, 0]
#         ciy = ci[:, 1]
#         ciz = ci[:, 2]
#         ax.scatter(cix, ciy, ciz, color=np.random.random(3))
#     plt.show()
