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
print(Y)

# plt.plot(np.arange(len(serie)), serie, label="g(n)", marker="o")
plt.plot(X, Y, label=f"Grado mayor: {iter}", alpha=0.6) 


plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(-rango - 1, rango + 1)
plt.ylim(-1.5, 1.5)
plt.legend()    
plt.show()