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

def descenso_gradiente(X,Y, alpha):
    var = [0,0,0]
    lista = [ ]
    mini = np.inf
    iteracion = [i for i in range(1500)]
    for i in range(1500):
        var = gradiente(X,Y,var.copy(), alpha) #Conseguimos recta para saber los puntos que están mas cercana
        mini = min(mini, coste(X,Y,var))
        lista.append(mini)
    plt.plot(iteracion, lista)
    return var, lista

def normalizar(X):
    media = np.mean(X)
    desviacion = np.std(X)
    X = (X - media) / desviacion
    mu = np.dot(X[:,1:], media)
    sigma = np.dot(X[:,1:], desviacion)
    return mu, sigma, X

def main():
    datos = carga_csv('ex1data2.csv')
    X = datos[:, :-1]
    Y = datos[:, -1]

    m = np.shape(X)[0]
    n = np.shape(X)[1]


    mu, sigma, nueva_X = normalizar(X)

    nueva_X = np.hstack([np.ones([m, 1]), nueva_X]) #añadimos una columna de 1's a la X
    
    alpha = 0.3
    Thetas, costes = descenso_gradiente(nueva_X, Y, alpha)

    alpha_list = [0.001,0.003,0.01,0.03,0.1,0.3]
    for alpha in alpha_list:
        thetas,costes = descenso_gradiente(nueva_X,Y,alpha)        

    plt.legend(alpha_list)
   
    plt.show()


main()
