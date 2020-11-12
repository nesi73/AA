#2.Deteccion de spam

from process_email import email2TokenList
from get_vocab_dict import getVocabDict
import codecs
from sklearn.svm import SVC
import numpy as np
import sys

def addX(directorio, X, Xval ,Xtest, numFicheros, dicVoc):
    for i in range(numFicheros):
        email_contents = codecs.open('{0}/{1:04d}.txt'.format(directorio,i+1), 'r', encoding='utf-8', errors='ignore').read()
        tokens  = email2TokenList(email_contents)
        arrayPalabras = np.zeros(1900)
        for palabra in tokens:
            if palabra in dicVoc:
                arrayPalabras[dicVoc[palabra]] = 1
        if(i + 1 <= int(numFicheros*0.7)):
            X = np.vstack((X,arrayPalabras)) #le añadimos a la fila correspondiente el array
        elif (i + 1 <= int(numFicheros*0.9)):
            Xval = np.vstack((Xval,arrayPalabras)) #le añadimos a la fila correspondiente el array
        else:
            Xtest = np.vstack((Xtest,arrayPalabras)) #le añadimos a la fila correspondiente el array
    return X,Xval,Xtest

def addY(numFicherosSpam, numFicherosNoSpam):
    y_ones = np.ones(int(numFicherosSpam))
    y_zeros = np.zeros(int(numFicherosNoSpam))
    return np.append(y_ones, y_zeros)

def main():
    np.set_printoptions(threshold=sys.maxsize)
    directorio = "spam"
    dicVoc = getVocabDict()
    num_spam = 500
    num_easy_ham = 2501
    num_hard_ham = 250
    X = np.empty((0,1900))
    Xval = np.empty((0,1900))
    Xtest = np.empty((0,1900))
    X, Xval, Xtest = addX("spam", X,Xval,Xtest, num_spam, dicVoc)
    X, Xval, Xtest = addX("easy_ham", X,Xval,Xtest, num_easy_ham, dicVoc)
    X, Xval, Xtest = addX("hard_ham", X,Xval,Xtest, num_hard_ham, dicVoc)

    print(np.shape(X))
    y = addY(num_spam*0.7, (num_easy_ham+num_hard_ham)*0.7)
    yval = addY(num_spam*0.2,(num_easy_ham+num_hard_ham)*0.2)
    ytest = addY(num_spam*0.1,(num_easy_ham+num_hard_ham)*0.1)

    print("X e Y añadido perfectamente")

    array = np.array([0.01,0.03,0.1,0.3,1,3,10,30])
    minimo = np.inf
    c,g = 0,0
    for i in array:
        print("C: ", i)
        for j in array:
            print("sigma: ", j)
            sigma = j
            svm = SVC(kernel='rbf', C=i, gamma=1/(2*sigma**2))
            svm.fit(X,y.ravel())
            pred = svm.predict(Xval)
            err = np.sum(pred != yval.ravel())/np.size(yval.ravel(), 0)
            if(minimo > err):
                minimo = err
                c = i
                g = j
    svm = SVC(kernel='rbf', C=c, gamma=1/(2*g**2))
    svm.fit(X,y.ravel())
    print(svm.predict(Xtest))
    err = np.sum(pred != ytest.ravel())/np.size(ytest.ravel(), 0)
    print(err)

main()
