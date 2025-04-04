import numpy as np
from mathkat import Funcion
from colorstreak import log
import random
import time
import  matplotlib.pyplot as plt
from tabulate import tabulate


def tiempo_ejec(func):
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        resultado = func(*args, **kwargs)
        fin = time.perf_counter()
        tiempo_ejecucion = round(fin - inicio,3)
        return resultado, tiempo_ejecucion
    return wrapper


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


@tiempo_ejec
def descenso_gradiente_esto_gen(X, y, w_init, gradiente_fn,alpha,batch_size,max_iter):
    if not isinstance(X, np.ndarray) or  not isinstance(y,np.ndarray):
        raise TypeError('Tus aprametros "x" y "y" ydebe ser un np.array')
    
    w = np.array(w_init,dtype=float)
    n = len(X)
    
    for epoch in range(max_iter):
        indices = np.random.permutation(n)
        
        # Procesamos el mini batch
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            x_batch = X[batch_indices]
            y_batch = y[batch_indices]
            grad,_ = gradiente_fn(x_batch, y_batch, w[0], w[1])
            w[0] -= alpha * grad[0]
            w[1] -= alpha * grad[1]
    return w
             



@tiempo_ejec
def gradiente_regreg_lineal_r2(x_v,y_v,w0,w1, batch = None):
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





# X,y = generador_datos_2d()
# X = np.array(X)
# y = np.array(y)
# #graficar_regresion(X,y,[3,5])


# grad_w, tiempo_grad = gradiente_regreg_lineal_r2(X, y, 3, 5)
# print(f"Gradiente calculado => grad_w0: {grad_w[0]:.4f}, grad_w1: {grad_w[1]:.4f} | Tiempo de cálculo: {tiempo_grad} seg")

# batches = [n+1 for n in range(100)]
# for batch in batches:
#     w,tiempo = descenso_gradiente_esto_gen(X, y, [3,5], gradiente_regreg_lineal_r2, alpha=0.01, batch_size=batch, max_iter=100) 
#     print(f"\nTamaño del batch {batch}")
#     print(f"Pesos finales obtenidos: w0 = {w[0]:.4f}, w1 = {w[1]:.4f} | Tiempo de ejecución: {tiempo} seg")
    
    
dataset_sizes = [100, 500, 1000]
batch_sizes   = [1, 10, 50]           
iterations    = [50, 100, 200]       

results = []
for ds in dataset_sizes:
    X_ds, y_ds = generador_datos_2d(m=ds)
    X_ds = np.array(X_ds)
    y_ds = np.array(y_ds)
    for bs in batch_sizes:
        for iters in iterations:
            w_final, tiempo_exec = descenso_gradiente_esto_gen(X_ds, y_ds, [3,5], gradiente_regreg_lineal_r2, alpha=0.01, batch_size=bs, max_iter=iters)
            results.append([ds, bs, iters, round(w_final[0], 4), round(w_final[1], 4), tiempo_exec])

headers = ["Dataset Size", "Batch Size", "Iterations", "w0", "w1", "Time (s)"]
print("\nResultados de Experimentos:")
print(tabulate(results, headers=headers, tablefmt="fancy_grid"))