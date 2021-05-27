import numpy as np
import gradientDescent as gd
import matplotlib.pyplot as plt


def plotPolinomial(coordenatesArray, thetaX, thetaY, thetaZ):
    ax = plt.axes(projection='3d')
    dataX = coordenatesArray[:, 0]
    dataY = coordenatesArray[:, 1]
    dataZ = coordenatesArray[:, 2]
    ax.scatter(dataX, dataY, dataZ, color=np.random.random(3))

    tempo = np.linspace(1/60, 1, 60)
    x = thetaX[0] + tempo * thetaX[1]
    y = thetaY[0] + tempo * thetaY[1]
    z = thetaZ[0] + tempo * thetaZ[1] + tempo**2 * thetaZ[2]  # + tempo**3 * thetaZ[3] + tempo**4 * thetaZ[4]
    ax.plot(x, y, z)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(0.0, 0.3)

    plt.show()


def runPolinomial(coordenatesArray, time):
    print("--------------------- treinando modelo polinomial ----------------------")
    thetaX = gd.initializeThetas(2, (-3.0, 3.0))
    thetaY = gd.initializeThetas(2, (0.0, 2.0))
    thetaZ = gd.initializeThetas(3, (0.0, 0.6))

    linearGD = gd.LinearRegression()
    polinomialGD = gd.PolinomialRegression()

    dataX = coordenatesArray[:, 0]
    dataY = coordenatesArray[:, 1]
    dataZ = coordenatesArray[:, 2]

    for i in range(0, 2000):
        thetaX = gd.runEpoch(linearGD, thetaX, time, dataX, 0.2)
        # print("Error X -> {}".format(gd.errorFunction(linearGD, thetaX, time, dataX, time.shape[0])))
        thetaY = gd.runEpoch(linearGD, thetaY, time, dataY, 0.2)
        # print("Error Y -> {}".format(gd.errorFunction(linearGD, thetaY, time, dataY, time.shape[0])))
        thetaZ = gd.runEpoch(polinomialGD, thetaZ, time, dataZ, 0.4)
        print("Error Z -> {}".format(gd.errorFunction(polinomialGD, thetaZ, time, dataZ, time.shape[0])))
        # print(" ----------------------------------------- ")

    plotPolinomial(coordenatesArray, thetaX, thetaY, thetaZ)


def plotLinear(coordenatesArray, thetaX, thetaY, thetaZ):
    ax = plt.axes(projection='3d')
    dataX = coordenatesArray[:, 0]
    dataY = coordenatesArray[:, 1]
    dataZ = coordenatesArray[:, 2]
    ax.scatter(dataX, dataY, dataZ, color=np.random.random(3))

    tempo = np.linspace(1/60, 1, 60)
    x = thetaX[0] + tempo * thetaX[1]
    y = thetaY[0] + tempo * thetaY[1]
    z = thetaZ[0] + tempo * thetaZ[1]
    ax.plot(x, y, z)

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(0.0, 0.3)

    plt.show()


def runLinear(coordenatesArray, time):
    print("--------------------- treinando modelo linear ----------------------")
    thetaX = gd.initializeThetas(2, (-3.0, 3.0))
    thetaY = gd.initializeThetas(2, (0.0, 2.0))
    thetaZ = gd.initializeThetas(2, (0.0, 0.6))

    linearGD = gd.LinearRegression()

    dataX = coordenatesArray[:, 0]
    dataY = coordenatesArray[:, 1]
    dataZ = coordenatesArray[:, 2]

    for i in range(0, 2000):
        thetaX = gd.runEpoch(linearGD, thetaX, time, dataX, 0.05)
        # print("Error X -> {}".format(gd.errorFunction(linearGD, thetaX, time, dataX, time.shape[0])))
        thetaY = gd.runEpoch(linearGD, thetaY, time, dataY, 0.05)
        # print("Error Y -> {}".format(gd.errorFunction(linearGD, thetaY, time, dataY, time.shape[0])))
        thetaZ = gd.runEpoch(linearGD, thetaZ, time, dataZ, 0.05)
        print("Error Z -> {}".format(gd.errorFunction(linearGD, thetaZ, time, dataZ, time.shape[0])))
        # print(" ----------------------------------------- ")

    plotLinear(coordenatesArray, thetaX, thetaY, thetaZ)


if __name__ == '__main__':
    data = np.genfromtxt('kick2.dat')
    # runLinear(data, np.linspace(1/60, 1/3, 20))
    runPolinomial(data, np.linspace(1/60, 1/3, 20))
