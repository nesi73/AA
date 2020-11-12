from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np 

def carga_csv(file_name):
    """carga"""
    valores = read_csv(file_name, header=None).values
    return valores.astype(float)

def gradiente(X,Y,Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X,Theta)
    Aux = (H - Y)
    for i in range(n):
        Aux_i = Aux * X[:, i]
        NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
    return NuevaTheta

def coste(X, Y, Theta):
    H = np.dot(X, Theta)
    Aux = (H - Y) ** 2
    return Aux.sum() / (2 * len(X))




def main():
    datos = carga_csv('ex1data1.csv')
    X = datos[:, :-1] #todas las filas y todas las columnas excepto la última columna esto en 2D
    Y = datos[:, -1] #todas las filas y la última columna esto en 1D

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.hstack([np.ones([m, 1]), X]) #añadimos una columna de 1's a la X

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    alpha = 0.01
    Thetas, costes = descenso_gradiente(X,Y, alpha)
    H = np.dot(X, Thetas)
    Theta0 = np.arange(-10,10,0.1)
    Theta1 = np.arange(-1,4,0.1)
    Theta0, Theta1 = np.meshgrid(Theta0,Theta1)
    C = np.empty_like(Theta0)
    for ix, iy in np.ndindex(Theta0.shape):
        C[ix, iy] = coste(X,Y, [Theta0[ix,iy],Theta1[ix,iy]])
    surf = ax.plot_surface(Theta0,Theta1, C, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.view_init(40, 35)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    fig.tight_layout()
    plt.show()

main()