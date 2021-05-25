import numpy as np
import gradientDescent as gd
import matplotlib.pyplot as plt


def plotLinear(dataXY, dataZ, tethas):
    ax = plt.axes(projection='3d')
    dataX = dataXY[:, 0]
    dataY = dataXY[:, 1]
    ax.scatter(dataX, dataY, dataZ, color=np.random.random(3))

    x = np.linspace(-3, 3, 100)
    y = np.linspace(0, 2, 100)
    finalFunction = tethas[0] + x * tethas[1] + y * tethas[2]
    ax.plot(x, y, finalFunction)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(0.0, 0.3)

    plt.show()


def plotPolinomial(dataXY, dataZ, tethas):
    ax = plt.axes(projection='3d')
    dataX = dataXY[:, 0]
    dataY = dataXY[:, 1]
    ax.scatter(dataX, dataY, dataZ, color=np.random.random(3))

    plt.xlim([-3.0, 3.0])
    plt.ylim([0.0, 2.0])

    x = np.linspace(-3, 3, 100)
    y = np.linspace(0, 2, 100)
    finalFunction = tethas[0] + x * tethas[1] + y ** 2 * tethas[2]
    ax.plot(x, y, finalFunction)
    ax.set_zlim((0.0, 0.3))
    plt.show()


def runPolinomial(entries, results):
    print("--------------------- treinando modelo polinomial ----------------------")
    tethas = gd.initializeTethas(2, [(-3.0, 3.0), (0.0, 2.0)])
    polinomialGD = gd.PolinomialRegression()
    for i in range(0, 2000):
        tethas = gd.runEpoch(polinomialGD, tethas, entries, results, 0.2)
        print(gd.errorFunction(polinomialGD, tethas, entries, results, entries.shape[0]))
    print("final tethas -->>> ")
    print(tethas)
    plotPolinomial(entries, results, tethas)


def runLinear(entries, results):
    print("--------------------- treinando modelo linear ----------------------")
    tethas = gd.initializeTethas(2, [(-3.0, 3.0), (0.0, 2.0)])
    linearGD = gd.LinearRegression()
    for i in range(0, 2000):
        tethas = gd.runEpoch(linearGD, tethas, entries, results, 0.2)
        print(gd.errorFunction(linearGD, tethas, entries, results, entries.shape[0]))
    print("final tethas -->>> ")
    print(tethas)
    plotLinear(entries, results, tethas)


if __name__ == '__main__':
    data = np.genfromtxt('kick1.dat')
    runLinear(data[:, :2], data[:, 2])
    runPolinomial(data[:, :2], data[:, 2])
