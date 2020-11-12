import displayData as dd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from checkNNGradients import checkNNGradients

def sigmoide(z):
    return (1/(1+np.exp(-z)))

def derivadaSigmoide(z):
    return np.dot(sigmoide(z),(1-sigmoide(z)).T) #(a2t * (1 - a2t)))

def pesosAleatorios(L_in, L_out):
    INIT_EPSILON = 0,12
    Theta1 = np.random.random((10,11))*(2*INIT_EPSILON) - INIT_EPSILON
    Theta2 = np.random.random((1,11))*(2*INIT_EPSILON) - INIT_EPSILON
    return Theta1,Theta2

def coste(a3,x, y, landa, theta1,theta2):
    m = len(x)
    cost = np.sum((y * np.log(a3)) + ((1-y) * np.log(1 - a3)))
    suma = (np.sum(np.power(theta1[:,1:],2))+ np.sum(np.power(theta2[:,1:],2)))
    regresion = (landa/(2*m))*suma
    return -(1/m)*cost + regresion


def forwardprop(theta1, theta2, x):
    a1 = x #activación capa 1 la misma que la entrada
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoide(z2)
    a2 = np.hstack([np.ones([np.shape(a2)[0],1]),a2])
    z3 = np.dot(a2,theta2.T)
    h = sigmoide(z3)
    print(derivadaSigmoide(z3))
    return a1, a2, h


def backprop(params_rn,num_entradas,num_ocultas, num_etiquetas,X,y,reg):
    theta1 = np.reshape(params_rn[:num_ocultas*(num_entradas + 1)],(num_ocultas, (num_entradas+1)))
    theta2 = np.reshape(params_rn[num_ocultas*(num_entradas + 1):],(num_etiquetas, (num_ocultas+1)))
    x_unos = np.hstack([np.ones([np.shape(X)[0],1]),X])
    m = x_unos.shape[0]
    DELTA1, DELTA2 = np.zeros(np.shape(theta1)), np.zeros(np.shape(theta2))
    a1, a2, h = forwardprop(theta1, theta2, x_unos) #hacemos la propagación hacia delante para conseguir las respectivas activaciones
    for t in range(m): #iteramos sobre cada uno de los casos prueba
        a1t = a1[t,:]
        a2t = a2[t,:]
        ht = h[t, :]
        yt = y[t]
        delta3 = ht - yt
        delta2 = (np.dot(theta2.T, delta3) * (a2t * (1 - a2t)))
        DELTA1 = DELTA1 + np.dot(delta2[1:, np.newaxis], a1t[np.newaxis,:])
        DELTA2 = DELTA2 + np.dot(delta3[:, np.newaxis], a2t[np.newaxis,:])
    gradientVec = np.concatenate((np.ravel(DELTA1),np.ravel(DELTA2))) 
    regularizacionG = gradientVec + (reg/m)*(np.sum(theta1[:,1:])+ np.sum(theta2[:,1:]))
    jval = coste(h,X,y, reg,theta1,theta2)
    print("COSTE: " ,jval)
    return jval,gradientVec

def main():
    data = loadmat('ex4data1.mat')
    #Se pueden consultar las claves con data.keys()
    y = data['y']
    X = data['X']
    #almacena los datos leidos en X,y
    #sample = np.random.choice(X.shape[0], 100)
    #fig, ax = dd.displayData(X[sample])
    #plt.axis('off')
    #plt.show()
    weights = loadmat('ex4weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    thetaVec = np.concatenate((np.ravel(theta1),np.ravel(theta2)))
    y=(y-1)
    m = len(X)
    y_onehot=np.zeros((m,10))#5000x10
    for i in range(m):
        y_onehot[i][y[i]]=1
    DELTA1, DELTA2 = backprop(thetaVec, 400,25,10,X,y_onehot,1)
    checkNNGradients(backprop, 1)
    #print("DELTA1: ", DELTA1, "\nDELTA2: ", DELTA2)

main()