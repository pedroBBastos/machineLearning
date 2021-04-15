import numpy as np
import matplotlib.pyplot as plt


def elbow_value(clustered_data, centroides):

    soma = 0
    for ponto in clustered_data:
        # obtendo as distancias para os centroides do ponto atual
        norms_centroides = np.linalg.norm(ponto[:2] - centroides, axis=1)
        soma += np.amin(norms_centroides)

    return soma


def plot_elbow_graphic(values, k):
    xpoints = np.array(k)
    ypoints = np.array(values)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cost Function J')
    plt.plot(xpoints, ypoints, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=7)
    plt.show()