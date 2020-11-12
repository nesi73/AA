from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

def sigmoide(z):
    return (1/(1+np.exp(-z)))

def coste(theta,X,Y, landa):
    Termino1 = np.dot(np.transpose(np.log(sigmoide(np.dot(X,theta)))), Y)
    Termino2 = np.dot(np.transpose(np.log(1 - sigmoide(np.dot(X,theta)))),(1 - Y))
    Termino3 = np.dot(landa / (2*len(X)), np.sum(np.power(theta,2)))
    return np.negative((Termino1 + Termino2) / len(X)) + Termino3

def gradiente(theta, X, Y, landa):
    return ((1/len(X)) *np.dot( np.transpose(X) , sigmoide(np.dot(X,theta)) - Y)) + np.dot(landa, theta)/len(X)

def oneVsAll(X, y, num_etiquetas, reg):
    """oneVsAll entrena varios clasificadores por regresión logísitica con término de regularización
    'reg' y devuelve el resultado en una matriz, donde la fila i-ésima corresponde al clasificador
    de la etiquéta i-ésima"""
    X = np.hstack([np.ones([np.shape(X)[0],1]),X])
    m,n = X.shape
    theta_opt = np.zeros((num_etiquetas, n))
    theta = np.zeros((n ,))
    for i in range (num_etiquetas):
        num = 10 if i == 0 else i
        result = fmin_tnc(func=coste, x0=theta, fprime=gradiente, args=(X,1*(y == num),reg))
        theta_opt[i,:] = result[0]
    return theta_opt 

def predictOneVsAll(theta_opt, X):
    X = np.hstack([np.ones([np.shape(X)[0],1]),X])
    p = sigmoide(X.dot(theta_opt.T))
    prob = np.argmax(p, axis=1)
    return prob

def main():
    data = loadmat('ex3data1.mat')
    #Se pueden consultar las claves con data.keys()
    y = data['y']
    X = data['X']
    #almacena los datos leidos en X,y

    num_etiquetas = 10
    landa = 0.1
    y = np.ravel(y)

    thetas_opt = oneVsAll(X,y,num_etiquetas, landa)
    q = predictOneVsAll(thetas_opt, X)
    prob = np.sum(q == (y%10))/np.size(y)
    print("Para un valor de 0,1 de regularizacion, el valor es: ", prob)

    #Selecciona aleatoriamente 10 ejemplos y los pinta

    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1,20).T)
    plt.axis('off')
    plt.show()

main()