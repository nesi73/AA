#Regresion logistica regularizada

import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from math import log10
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

def carga_cvs(file_name):
    v = read_csv(file_name,header=None).values
    return v

def sigmoide(z):
    return (1/(1+np.exp(-z)))

def coste(theta,X,Y, landa):
    Termino1 = np.dot(np.transpose(np.log(sigmoide(np.dot(X,theta)))), Y)
    Termino2 = np.dot(np.transpose(np.log(1 - sigmoide(np.dot(X,theta)))),(1 - Y))
    Termino3 = np.dot(landa / (2*len(X)), np.sum(np.power(theta,2)))
    return np.negative((Termino1 + Termino2) / len(X)) + Termino3

def gradiente(theta, X, Y, landa):
    return ((1/len(X)) *np.dot( np.transpose(X) , sigmoide(np.dot(X,theta)) - Y)) + np.dot(landa, theta)/len(X)

def pintar(X, Y, theta, poly):
    plt.figure()

    #visualización de los datos
    pos = np.where(Y == 1) #Obtiene un vector con los índices de los ejemplos positivos
    pos2 = np.where(Y == 0)

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k') ## Dibuja los ejemplos positivos
    plt.scatter(X[pos2,0],X[pos2,1], color='green')

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1,xx2 = np.meshgrid(np.linspace(x1_min, x1_max),np.linspace(x2_min, x2_max))

    a = [1,2,3]
    b = [4,5,6]
    c = np.c_[a,b]
    c = poly.fit_transform(c)
    h= sigmoide(poly.fit_transform(np.c_[xx1.ravel(),xx2.ravel()]).dot(theta))

    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig("frontera3.pdf")
    plt.close()

def main():
    D = carga_cvs("ex2data2.csv")
    X = D[:,:-1] #shape (100,2)
    Y = D[:,2]  #shape (100,)

    poly = PolynomialFeatures(6)
    X = poly.fit_transform(X)

    theta = np.zeros(np.shape(X)[1])
    landa = 0.5

    result = opt.fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(X,Y,landa))
    theta_opt = result[0]
    pintar(X[:,1:],Y, theta_opt, poly) 

main()