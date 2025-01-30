import matplotlib.pyplot as plt
import numpy as np
import math as mt
import time




def g(n):
    return (4*(n**2)) + (7*n)


def f(n,c=1):
    return n*n


def g_1(n):
    return n


def f_1(n,c=1):
    return mt.log10(n)



def graficador(g,f,n_max=10):
    n_valores = np.arange(1,n_max+1)
    tiempo_inicio = time.perf_counter()
    x = np.array([g(n) for n in n_valores])
    y = np.array([f(n) for n in n_valores])
    tiempo_fin = time.perf_counter()
    total = tiempo_fin- tiempo_inicio
    print(f"Tiempo del método iterativo: {(total) * 1_000_000:.2f} microsegundos")
    plt.plot(n_valores, x, label="g(n)", marker="o")
    plt.plot(n_valores, y, label="f(n)", marker="s")

    plt.xlabel("n")
    plt.ylabel("Valores de funciones")
    plt.title("Comparación de funciones g(n) y f(n)")
    plt.legend()
    plt.grid(True)
    plt.show()


graficador(g,f)


graficador(g_1,f_1)