from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
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
    return Aux.sum() / (2 * len(X))Aux.sum()

def descenso_gradiente(X,Y, alpha):
    var = [0,0]
    mini = np.inf
    for i in range(1500):
        var = gradiente(X,Y,var.copy(), alpha) #Conseguimos recta para saber los puntos que están mas cercana
        mini = min(mini, coste(X,Y,var))
    return var, mini


def main():
    datos = carga_csv('ex1data1.csv')
    X = datos[:, :-1] #todas las filas y todas las columnas excepto la última columna esto en 2D
    Y = datos[:, -1] #todas las filas y la última columna esto en 1D
    plt.scatter(X,Y,marker='x')

    m = np.shape(X)[0]
    n = np.shape(X)[1]

    X = np.hstack([np.ones([m, 1]), X]) #añadimos una columna de 1's a la X

    alpha = 0.01
    Thetas, costes = descenso_gradiente(X,Y, alpha)
    H = np.dot(X, Thetas)
    plt.plot(X[:,-1], H, color='red')
    plt.show()

main()