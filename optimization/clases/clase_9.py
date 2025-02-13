import matplotlib.pyplot as plt
import math as mt
import numpy as np




def taylor_2(x,lim):
    sum = x
    signo = -1
    for n in range(3, lim+1, 2):
        sum += signo * ((1/mt.factorial(n)) * x**n)
        signo *= -1
    return sum


rango = 50
lim = 51

X = np.linspace(-rango, rango,1000)
Y = [taylor_2(x, lim) for x in X]

# plt.plot(np.arange(len(serie)), serie, label="g(n)", marker="o")
plt.plot(X, Y, label="Taylor series approximation")
plt.xlabel("n")
plt.ylabel("Valores de funciones")
plt.title("Comparación de funciones g(n) y f(n)")
plt.xlim(-rango - 1, rango + 1)
plt.legend()
plt.grid(True)
plt.show()