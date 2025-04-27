
import numpy as np




# Ejemplo de uso
f = lambda x, y: (x-1)**2 + (y-2)**2
grad_f = lambda x, y: np.array([2*(x-1), 2*(y-2)])
x_0 = np.array([0.0, 0.0])
v_0 = np.array([0.0, 0.0])
alpha = 0.1
iteraciones = 10
epsilon = 1e-6
eta = 0.9

funcion = Gradiente(f, grad_f, x_0, v_0, alpha, iteraciones, epsilon, eta)
funcion.simple()
funcion.momentum()
funcion.nesterov()