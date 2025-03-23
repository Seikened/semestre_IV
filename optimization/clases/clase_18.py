import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate



class Funcion:
    def __init__(self,f,grad_f,x_0,v_0,alpha,iteraciones,epsilon,eta):
        self.f = f
        self.grad_f = grad_f
        self.x_0 = x_0
        self.v_0 = v_0
        self.alpha = alpha
        self.iteraciones = iteraciones
        self.epsilon = epsilon
        self.eta = eta
        self.x_historico = [x_0]
        self.headers = ["Iteración", "x", "Norma"]
        self.data_grad_simple = []
        self.data_grad_momentum = []

    
    def imprimir_tabla_tabulate(self,data, headers):
        """
        Imprime una tabla formateada usando la librería tabulate.

        :param headers: Lista con los nombres de las columnas.
        :param data: Lista de filas, donde cada fila es una lista o tupla de valores.
        """
        print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

    def desenso_gradiente_simple(self):
        f = self.f
        grad_f = self.grad_f
        x0 = self.x_0
        lr = self.alpha
        max_iters = self.iteraciones
        epsilon = self.epsilon
        x_historico = [x0]
        
        for i in range(max_iters):
            f_i = f(*x0)
            grad_f_i = grad_f(*x0)
            nomra_grad = np.linalg.norm(grad_f_i)
            if nomra_grad < epsilon: # Criterio de paro 
                break
            xi = x0 - lr * grad_f_i
            x0 = xi.copy()
            x_historico.append(x0)
            self.data_grad_simple.append((i+1, x0.tolist(), nomra_grad))
            
        self.imprimir_tabla_tabulate(self.data_grad_simple, self.headers)
        return x_historico


    def desenso_gradiente_momentum(self):
        f = self.f
        grad_f = self.grad_f
        x0 = self.x_0
        v0 = self.v_0
        lr = self.alpha
        eta = self.eta
        max_iters = self.iteraciones
        epsilon = self.epsilon
        x_historico = [x0]
        
        for i in range(max_iters):
            f_i = f(*x0)
            grad_f_i = grad_f(*x0)
            nomra_grad = np.linalg.norm(grad_f_i)
            if nomra_grad < epsilon: # Criterio de paro 
                break
            vi = eta * v0 + lr * grad_f_i
            xi = x0 - vi
            x0 = xi.copy()
            v0 = vi.copy()
            x_historico.append(x0)
            self.data_grad_momentum.append((i+1, x0.tolist(), nomra_grad))
            
        self.imprimir_tabla_tabulate(self.data_grad_momentum, ["Iteración", "x", "Norma", "velocidad"])
        return x_historico
        
# ===================== Funciones =====================


f1 = lambda x: x**2 + 2*x + 1
grad_f1 = lambda x: np.array([2*x + 2])

f2 = lambda x1,x2: x1**2 + (2*x2**2)
grad_f2 = lambda x1,x2: np.array([2*x1, 4*x2])

f3 = lambda x1,x2,x3: x1**2 + x2**2 + 2*x3**2
grad_f3 = lambda x1,x2,x3: np.array([2*x1, 2*x2, 4*x3])

# ================= Función 1 =================

x_0 = np.array([10])
alpha = 0.1
v_0 = np.array([0])
iteraciones = 50
epsilon = 0.001
eta = 0.9

f1 = Funcion(f1,grad_f1,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION SIN MOMENTUM")
f1.desenso_gradiente_simple()
print("Momentum F1")
f1.desenso_gradiente_momentum()

# # ================= Función 2 =================

x_0 = np.array([-5,-2])
alpha = 0.1
v_0 = np.array([0,0])
iteraciones = 50
epsilon = 0.001
eta = 0.9

f2 = Funcion(f2,grad_f2,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION SIN MOMENTUM")
f2.desenso_gradiente_simple()
print("Momentum F2")
f2.desenso_gradiente_momentum()

# # ================= Función 3 =================
x_0 = np.array([-1,-1,-1])
alpha = 0.1
v_0 = np.array([0,0,0])
iteraciones = 50
epsilon = 0.001
eta = 0.9
f3 = Funcion(f3,grad_f3,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION SIN MOMENTUM")
f3.desenso_gradiente_simple()
print("Momentum F3")
f3.desenso_gradiente_momentum()