import numpy as np
import math as math
import abc


# inicializa randomicamente os tethas a serem utilizados
def initializeTethas(nFeatures, featuresRangeArray):
    tethas = np.zeros((nFeatures + 1))
    for i in range(len(featuresRangeArray)):
        tethas[i + 1] = np.random.uniform(low=featuresRangeArray[i][0], high=featuresRangeArray[i][1])
    return tethas


def updateTetha(baseGradientDescent, tethas, currentTetha, learningRate, entries, results, m):
    somatorio = 0
    for i in range(m):
        ithX = 1 if currentTetha == 0 else entries[i][currentTetha - 1]
        somatorio += (baseGradientDescent.calculateHypothesisForPoint(tethas, entries[i]) - results[i]) * ithX
    return tethas[currentTetha] - learningRate * (somatorio / m)


# tethas -> vetor de tethas atual (n features + 1)
# entries -> vetor com m entradas de n features
# results -> vetor com m resultados
# learningRate -> learning rate
def runEpoch(baseGradientDescent, tethas, entries, results, learningRate):
    m = entries.shape[0]
    updatedTethas = np.zeros(tethas.size)
    for ithTetha in range(tethas.size):
        updatedTethas[ithTetha] = updateTetha(baseGradientDescent, tethas, ithTetha, learningRate, entries, results, m)
    return updatedTethas


def errorFunction(baseGradientDescent, tethas, entries, results, m):
    somatorio = 0
    for i in range(m):
        somatorio += math.pow((baseGradientDescent.calculateHypothesisForPoint(tethas, entries[i]) - results[i]), 2)
    return somatorio / (2 * m)


class BaseGradientDescent(metaclass=abc.ABCMeta):
    # calcula hTetha(xi), sendo xi o vetor com os dados de uma entrada ("linha")
    @abc.abstractmethod
    def calculateHypothesisForPoint(self, tethas, entry):
        pass


class LinearRegression(BaseGradientDescent):
    def calculateHypothesisForPoint(self, tethas, entry):
        hyphotesis = tethas[0]
        for i in range(entry.size):
            hyphotesis += entry[i] * tethas[i + 1]
        return hyphotesis


class PolinomialRegression(BaseGradientDescent):
    def calculateHypothesisForPoint(self, tethas, entry):
        hyphotesis = tethas[0]
        for i in range(entry.size):
            hyphotesis += tethas[i + 1] * math.pow(entry[i], i + 1)  # ^1, ^2, ^3...
        return hyphotesis
