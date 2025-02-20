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



def e_x(x,lim,a=0):
    sum = 0
    for n in range(0, lim+1):
        sum += x**n / mt.factorial(n)
    return sum



def funcion_cualquiera(x,grado,a=0):
    sum = x
    if grado == 0:
        return (3*x**2) + (4*x**2) - (2*x) + 1 
    if grado == 1:
        return (9*x**2) + (8*x**2) - 2 
    if grado == 2:
        return (18*x) + 8
    if grado == 3:
        return 18
    if grado == 4:  
        return 0
    





rango = 50
lim = 5

X = np.linspace(-rango, rango,1000)
Y = [taylor_2(x, lim) for x in X]
z = [e_x(x, lim) for x in X]

no_n_diferenciables = [funcion_cualquiera(x, 4) for x in X]
error = [abs(mt.sin(x) - taylor_2(x, lim)) for x in X]



plt.plot(X, Y, label="Taylor approximation", alpha=0.6)
plt.plot(X, [mt.sin(x) for x in X], label="sin(x)", alpha=0.6)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(-rango - 1, rango + 1)
plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()


plt.plot(X, error, label="Error", color="red", alpha=0.6)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(-rango - 1, rango + 1)
plt.legend()

plt.show()



plt.plot(X, z, label="Taylor approximation", alpha=0.6)
plt.plot(X, [mt.exp(x) for x in X], label="exp(x)", alpha=0.6)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(-rango - 1, rango + 1)
plt.ylim(-1.5, 1.5)
plt.legend()

plt.show()


plt.plot(X, no_n_diferenciables, label="Taylor approximation", alpha=0.6)
plt.plot(X, [funcion_cualquiera(x, 0) for x in X], label="f(x)", alpha=0.6)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(-rango - 1, rango + 1)
plt.ylim(-1.5, 1.5)
plt.legend()


plt.show()