import numpy as np
from mathkat import Funcion
from colorstreak import log
import random
import  matplotlib.pyplot as plt





def generador_datos_2d(m=100,w_verdadero=[3,5],x_limit=10, ruido_std=1.0):
    x = np.random.rand(m) * x_limit
    ruido = np.random.randn(m) * ruido_std
    y = (w_verdadero[0] + w_verdadero[1] * x) + ruido
    return x,y


def graficar_regresion(x,y,w):
    plt.scatter(x,y,color="m",marker="o",s=30)
    y_pred = w[0] + w[1] * x
    plt.plot(x,y_pred,color="g")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def descenso_gradiente_esto_gen(X, y, w_init, gradiente_fn,alpha,batch_size,max_iter):
    if not isinstance(X, np.ndarray) or  not isinstance(y,np.ndarray):
        raise TypeError('Tus aprametros "x" y "y" ydebe ser un np.array')
    w = w_init
    x_lista = []
    x = np.arrange(1,len(X),1)
    for i in range(max_iter):
        x = random.shuffle(X)
        x = x[:batch_size+1].sort()
        
        
        x_batch = None
        y_batch = None
        for j in range(len(x_batch)):
            x_batch = x[i:j + batch_size]
            y_batch = y[i:j + batch_size]
            g = gradiente_fn(x_batch,y_batch,w)
            w = w - alpha * g
    return w




def gradiente_regreg_lineal_r2(x_v,y_v,w0,w1, batch = None):
    #x_v = np.array([x_v])
    #y_v = np.array([y_v])
    w = np.array([w0,w1])
    if not isinstance(x_v, np.ndarray) or  not isinstance(y_v,np.ndarray):
        raise TypeError("Tus aprametros debe ser un np.array")
    
    suma_e = 0
    suma_e_x  = 0
    b = len(x_v) if not batch else batch
    for i in range(b):
        y_gorro = w[1] * x_v[i] + w[0]
        error = y_v[i] - y_gorro
        suma_e = suma_e - 2 * error
        suma_e_x = suma_e_x - 2 * error * x_v[i]
    grad_w0 = suma_e/b
    grad_w1 = suma_e_x/b
    return grad_w0,grad_w1





X,y = generador_datos_2d()
X = np.array(X)
y = np.array(y)
#graficar_regresion(X,y,[3,5])



grad_w0,grad_w1 = gradiente_regreg_lineal_r2(X,y,3,5)
print(grad_w0,grad_w1)




#w = descenso_gradiente_esto_gen(X,y,[3,5])