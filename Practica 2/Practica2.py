#Regresion logistica

import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from math import log10
import scipy.optimize as opt

def carga_cvs(file_name):
    v = read_csv(file_name,header=None).values
    return v

def sigmoide(z):
    return (1/(1+np.exp(-z)))

def coste(theta,X,Y):
    Termino1 = np.dot(np.transpose(np.log(sigmoide(np.dot(X,theta)))), Y)
    Termino2 = np.dot(np.transpose(np.log(1 - sigmoide(np.dot(X,theta)))),(1 - Y))
    return np.negative((Termino1 + Termino2) / len(X))

def gradiente(theta, X, Y):
    return (1/len(X)) *np.dot( np.transpose(X) , sigmoide(np.dot(X,theta)) - Y)

def pintar(X, Y, theta):
    plt.figure()

    #visualización de los datos
    pos = np.where(Y == 1) #Obtiene un vector con los índices de los ejemplos positivos
    pos2 = np.where(Y == 0)

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k') ## Dibuja los ejemplos positivos
    plt.scatter(X[pos2,0],X[pos2,1], color='green')

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1,xx2 = np.meshgrid(np.linspace(x1_min, x1_max), 
                          np.linspace(x2_min, x2_max))

    h= sigmoide(np.c_[np.ones((xx1.ravel().shape[0],1)),
                     xx1.ravel(),
                  5   xx2.ravel()].dot(theta))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("frontera.pdf")
    plt.close()

def main():
    D = carga_cvs("ex2data1.csv")
    X = D[:,:-1] #shape (100,2)
    Y = D[:,2]  #shape (100,)

    theta = np.zeros(3)

    X = np.hstack([np.ones([np.shape(X)[0],1]),X])

    print(coste(theta,X,Y))
    print(gradiente(theta,X,Y))

    result = opt.fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(X,Y))
    theta_opt = result[0]
    sol = coste(theta_opt, X, Y)
    pintar(X[:,1:],Y, theta_opt)    
    #visualización de los datos
    pos = np.where(Y == 1) #Obtiene un vector con los índices de los ejemplos positivos
    pos2 = np.where(Y == 0)

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k') ## Dibuja los ejemplos positivos
    plt.scatter(X[pos2,0],X[pos2,1], color='green')

main()
