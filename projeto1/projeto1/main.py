import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt

from sklearn import preprocessing
import sklearn.metrics as sklMetrics

import kMeans as km
import dbscan as dbscan
import split_train_test as stt
import elbow as em
import scipy.spatial


def executeDBSCAN(dataTraining, dataTest):
    min_max_scaler = preprocessing.MinMaxScaler()
    dataTraining = min_max_scaler.fit_transform(dataTraining)

    matrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(dataTraining))

    # knee gráfico
    thirdMinDists = np.sort(np.partition(matrixDeDistancia, 4)[:, 4])
    em.plot_elbow_graphic(thirdMinDists, range(0, len(thirdMinDists)), 2)

    clusteredPorEps = []
    corePointsPorEps = []
    clusterNumbersPorEps = []

    eps_values = [.058] # eps_values = np.linspace(0.0, 0.5)
    for eps in eps_values:
        clustered, corePointsList = dbscan.DBSCAN(dataTraining, matrixDeDistancia, eps, 3)
        clusterNumbers = np.unique(clustered[:, 3])

        clusteredPorEps.append(clustered)
        corePointsPorEps.append(corePointsList)
        clusterNumbersPorEps.append(clusterNumbers)

        # silhouette_score como métrica para avaliar
        silhouette_score = sklMetrics.silhouette_score(dataTraining, clustered[:, 3])

        for ci in clusterNumbers:
            ci = clustered[clustered[:, 3] == ci]
            ci = ci[:, :2]
            cix = ci[:, 0]
            ciy = ci[:, 1]
            plt.plot(cix, ciy, color=np.random.random(3), marker='x', linestyle='')
        plt.show()

    ###########################################################
    # Test
    ###########################################################

    dataTest = min_max_scaler.fit_transform(dataTest)
    pontos, dimensoes = dataTest.shape

    chosenClustered = clusteredPorEps[0]
    chosenClustered = np.delete(chosenClustered, 2, 1)
    initCluster = -1 * np.ones((pontos, 1))  # -1 (no cluster)
    dataTest = np.hstack((dataTest, initCluster))

    newMatrix = np.concatenate((chosenClustered, dataTest))
    novaMatrixDeDistancia = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(newMatrix))
    novaMatrixDeDistancia = novaMatrixDeDistancia[516:, :516]
    print(novaMatrixDeDistancia)

    indicesPontosMaisProximos = np.argmin(novaMatrixDeDistancia, axis=1)
    # TODO Verificar se linha abaixo está correta para pegar as menores
    #  distancias de cada ponto novo
    menoresDistancias = novaMatrixDeDistancia[:, indicesPontosMaisProximos][:, 0]

    corePointsPorEps = corePointsPorEps[0]
    for i in range(0, len(menoresDistancias)):
        if menoresDistancias[i] <= eps_values[0]:
            index = 0
            while True:
                if corePointsPorEps[index][0] == indicesPontosMaisProximos[i]:
                    dataTest[i, 2] = corePointsPorEps[index][1]
                    break
    print(dataTest)



def executeKMeans(dataTraining, dataTest):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(dataTraining)

    centroidesPorK = []
    clusteredPorK = []
    clusteredNumbersPorK = []
    k_clusters = [1, 2, 3, 4, 5, 6, 7]
    elbow_values_plot = []

    for k in k_clusters:
        clustered, centroides = km.kMeans(data, k)
        clusterNumbers = np.unique(clustered[:, 2])

        centroidesPorK.append(centroides)
        clusteredPorK.append(clustered)
        clusteredNumbersPorK.append(clusterNumbers)

        for ci in clusterNumbers:
            ci = clustered[clustered[:, 2] == ci]
            ci = ci[:, :2]
            cix = ci[:, 0]
            ciy = ci[:, 1]
            plt.plot(cix, ciy, color=np.random.random(3), marker='x', linestyle='')
        plt.show()

        value = em.elbow_value(clustered, centroides)
        elbow_values_plot.append(value)

    em.plot_elbow_graphic(elbow_values_plot, k_clusters, 1)

    ################################################################
    # melhor k = 3
    # verificando para qual cluster pertencem os dados de teste
    ################################################################

    dataTest = min_max_scaler.fit_transform(dataTest)

    linhas = dataTest.shape[0]
    colunaCluster = -1 * np.zeros((linhas, 1))
    dataTest = np.hstack((dataTest, colunaCluster))

    centroidesTeste = centroidesPorK[2]
    clusteredTeste = clusteredPorK[2]
    clusterNumbersTeste = clusteredNumbersPorK[2]

    for ponto in dataTest:
        # obtendo as distancias para os centroides do ponto atual
        normsFromCentroides = np.linalg.norm(ponto[:2] - centroidesTeste, axis=1)
        # computando o centroide mais próximo
        nearestCentroideIndex = np.where(normsFromCentroides == np.amin(normsFromCentroides))[0][0]
        # indicando no ponto o indice do centroide ao qual ele pertence
        ponto[2] = nearestCentroideIndex

    # np.append(clusteredTeste, dataTest[:, :2])

    for ci in clusterNumbersTeste:
        cor = np.random.random(3)

        pontosCi = clusteredTeste[clusteredTeste[:, 2] == ci]
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


if __name__ == '__main__':
    data = np.genfromtxt('cluster.dat')
    training_set, test_set = stt.split_train_test(data, 0.1)
    # executeKMeans(training_set, test_set)
    executeDBSCAN(training_set, test_set)
    # executeKMeans(test_set)
    # executeDBSCAN(test_set)
