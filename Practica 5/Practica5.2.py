# 5.2 Curvas de aprendizaje

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt


def coste(theta,X,y, landa):
    theta = theta.reshape(-1,1)
    m = y.size
    h = X.dot(theta)
    Termino1 = (1/(2*m))*(np.sum(np.square(h - y)))
    Termino2 = (landa / (2*m))
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
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    X_ones = np.hstack([np.ones([np.shape(X)[0],1]),X])
    Xval_ones = np.hstack([np.ones([np.shape(Xval)[0],1]),Xval])
    m = len(X)
    Train_error = np.zeros(m)
    Validation_error = np.zeros(m)
    for i in range(m):
        result = opt.minimize(coste, theta,args=(X_ones[0:i+1],y[0:i+1],0), method = None, jac=gradiente, options={'maxiter':5000})
        Train_error[i] = result.fun
        Validation_error[i] = coste(result.x, Xval_ones, yval, landa)
    plt.plot(np.arange(0,12), Validation_error, color='orange',label='Cross Validation')
    plt.plot(np.arange(0,12),Train_error, color='purple', label='Train')
    plt.legend()
    plt.show()  
main()
