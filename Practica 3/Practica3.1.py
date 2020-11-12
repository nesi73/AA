from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

def sigmoide(z):
    return (1/(1+np.exp(-z)))

def predict(theta1, theta2, X):
    X = np.hstack([np.ones([np.shape(X)[0],1]),X])
    a = sigmoide(X.dot(theta1.T))
    a = np.hstack([np.ones([np.shape(X)[0],1]),a])
    a_2 = sigmoide(a.dot(theta2.T))
    prob = np.argmax(a_2, axis = 1) + 1
    return prob

def main():
    data = loadmat('ex3data1.mat')
    #Se pueden consultar las claves con data.keys()
    y = data['y']
    X = data['X']
    #almacena los datos leidos en X,y

    weights = loadmat('ex3weights.mat')
    theta1,theta2 = weights['Theta1'], weights['Theta2']

    q = predict(theta1,theta2, X)
    y = np.ravel(y)
    prob = np.sum(q == y)/np.size(y)
    print("La precision para la red neuronal esta en torno: ", prob)

main()