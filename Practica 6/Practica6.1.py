#Kernel lineal
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

def paint(X, y, svm):
    #visualización de los datos
    pos = (y == 1).ravel() #Obtiene un vector con los índices de los ejemplos positivos
    pos2 = (y == 0).ravel()

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k') ## Dibuja los ejemplos positivos
    plt.scatter(X[pos2,0],X[pos2,1], color='green')

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    x1,x2 = np.meshgrid(np.linspace(x1_min, x1_max,len(X)), np.linspace(x2_min, x2_max,len(X)))
    yp = svm.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)

    plt.contour(x1,x2, yp)
    plt.show()

def main():
    data = loadmat('ex6data1.mat')
    y = data['y']
    X = data['X']
    print(X)
    print(y)
    svm = SVC(kernel='linear', C=100)
    svm.fit(X,y.ravel())
    paint(X,y,svm)

main()
