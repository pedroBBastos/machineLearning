import numpy as np
import math as math
import abc


# inicializa randomicamente os thetas a serem utilizados
def initializeThetas(qtdThetas, rangeTuple):
    thetas = np.zeros(qtdThetas)
    for i in range(qtdThetas):
        thetas[i] = np.random.uniform(low=rangeTuple[0], high=rangeTuple[1])
    return thetas


def updatetheta(baseGradientDescent, thetas, currenttheta, learningRate, entries, results, m):
    somatorio = 0
    for i in range(m):
        ithX = 1 if currenttheta == 0 else entries[i]
        somatorio += (baseGradientDescent.calculateHypothesisForPoint(thetas, entries[i]) - results[i]) * ithX
    return thetas[currenttheta] - learningRate * (somatorio / m)


# thetas -> vetor de thetas atual (n features + 1)
# entries -> vetor com m entradas de n features
# results -> vetor com m resultados
# learningRate -> learning rate
def runEpoch(baseGradientDescent, thetas, entries, results, learningRate):
    m = entries.shape[0]
    updatedthetas = np.zeros(thetas.size)
    for iththeta in range(thetas.size):
        updatedthetas[iththeta] = updatetheta(baseGradientDescent, thetas, iththeta, learningRate, entries, results, m)
    return updatedthetas


def errorFunction(baseGradientDescent, thetas, entries, results, m):
    somatorio = 0
    for i in range(m):
        somatorio += math.pow((baseGradientDescent.calculateHypothesisForPoint(thetas, entries[i]) - results[i]), 2)
    return somatorio / (2 * m)


class BaseGradientDescent(metaclass=abc.ABCMeta):
    # calcula htheta(xi), sendo xi o vetor com os dados de uma entrada ("linha")
    @abc.abstractmethod
    def calculateHypothesisForPoint(self, thetas, entry):
        pass


class LinearRegression(BaseGradientDescent):
    def calculateHypothesisForPoint(self, thetas, entry):
        return thetas[0] + entry * thetas[1]


class PolinomialRegression(BaseGradientDescent):
    def calculateHypothesisForPoint(self, thetas, entry):
        hyphotesis = thetas[0]
        for i in range(thetas.size-1):
            hyphotesis += thetas[i + 1] * math.pow(entry, i + 1)  # ^1, ^2, ^3...
        return hyphotesis
