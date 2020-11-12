# 5.1 Regresión lineal regularizada

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt


def coste(theta,X,y, landa):
    theta = theta.reshape(-1,1)
    m = y.size
    h = X.dot(theta)
    Termino1 = (1/(2*m))*(np.sum(np.square(h - y)))
    Termino2 = (landa / (2*m)) * ((np.sum(np.square(theta[1:]))))
    J = Termino1 + Termino2
    return J

def gradiente(theta, X, Y, landa):
    theta = theta.reshape(-1,1)
    grad = ((1/len(X)) *np.dot( X.T , np.dot(X,theta) - Y)) + (landa/len(X)) * theta
    return grad.flatten()

def main():
    data = loadmat('ex5data1.mat')    
    y = data['y']
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']
    Xtest = data['Xtest']
    ytest = data['ytest']
    theta = np.ones((2,1))
    landa = 1
    plt.scatter(X,y, c='pink', marker='x')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    X_ones = np.hstack([np.ones([np.shape(X)[0],1]),X])
    print(coste(theta,X_ones,y,landa))
    print(gradiente(theta,X_ones,y, landa))   
    result = opt.minimize(coste, theta,args=(X_ones,y,0), method = None, jac=gradiente, options={'maxiter':5000})
    H = np.dot(X_ones, result.x)
    plt.plot(X,H, color='purple')
    plt.show()  
main()
