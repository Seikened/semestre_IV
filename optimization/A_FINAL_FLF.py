from mathkat import Gradiente
import numpy as np
import autograd.numpy as np
from autograd import elementwise_grad, hessian
from colorstreak import log



# ========== Examen ==========
func = lambda x, y: (((x**2) + y - 11)**2) + ((x + (y**2) - 7)**2)

rango = 4
n_numeros = 100
x = np.linspace(-rango, rango, n_numeros)
y = np.linspace(-rango, rango, n_numeros)

# Primeras derivadas parciales
f_dev_x = elementwise_grad(func)  
f_dev_y = elementwise_grad(func)


# Segundas derivadas parciales
f2_dev_x = elementwise_grad(f_dev_x)
f2_dev_y = elementwise_grad(f_dev_y)


# Hessian
hess_f = hessian(func, argnum=(0, 1))



def minimo_cond(punto):
    if punto > 0:
        return "min"
    else:
        return "max"



minimos = []

for x,y in zip(x,y):
    
    x = f2_dev_x(x,y)
    y = f2_dev_y(x,y)
    
    if minimo_cond(x) == "min":
        minimos.append([x,y])

print(f"Se obtuvieron {len(minimos)}")
for x,y in minimos:
    log.warning(f"Mínimo en  X:{x:1f} | Y:{y:1f}")