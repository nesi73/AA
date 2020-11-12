import matplotlib.pyplot as plt 
import numpy as np
import time

def integra_mc(fun, a, b, num_puntos=10000):
    cont = 0
    tic = time.process_time()
    x = (b - a) * np.random.random(num_puntos) + a
    y = np.random.random(num_puntos)
    for i in range(num_puntos):
       if(y[i] < fun(x[i])):
            cont += 1
    toc = time.process_time()
    return 1000 * (toc - tic)

def integra_mc_fast(fun, a, b, num_puntos=10000):
    tic = time.process_time()
    x = (b - a) * np.random.random(num_puntos) + a
    y = np.random.random (num_puntos)
    np.sum(fun(x) > y)
    toc = time.process_time()
    return 1000 * (toc - tic)
    
def main():
    X = np.linspace(0, np.pi, 256, endpoint=True)
    S = np.sin(X)
    times = []
    timesFast = []
    num_puntos = 10000
    sizes = np.linspace(100, 10000, 20)
    #plt.plot(X, S, color="blue") funcion sen pintada  
    for size in sizes:  
        times += [integra_mc(np.sin,0, np.pi, int(size))]
        timesFast += [integra_mc_fast(np.sin,0, np.pi, int(size))]
    plt.figure()
    plt.scatter(sizes, times, c='red', label='bucle')
    plt.scatter(sizes, timesFast, c='blue', label='vector')
    plt.legend()


main()
plt.show()
