import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate



class Funcion:
    """
    Clase que representa una función objetivo y sus métodos de optimización mediante descenso de gradiente.
    Permite aplicar diferentes variantes del descenso de gradiente y visualizar los resultados en tablas.
    """
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

        Parámetros:
        - data: Lista de filas, donde cada fila es una lista o tupla de valores a mostrar.
        - headers: Lista con los nombres de las columnas de la tabla.
        """
        print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

    def descenso_gradiente_simple(self):
        """
        Realiza el descenso de gradiente estándar para minimizar la función objetivo.
        En cada iteración, actualiza la posición usando el gradiente y almacena el historial de posiciones y normas.
        Imprime una tabla con los resultados de cada iteración.
        
        Retorna:
        - x_historico: Lista con el historial de posiciones x en cada iteración.
        """
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


    def descenso_gradiente_momentum(self):
        """
        Aplica el método de descenso de gradiente con momentum para minimizar la función objetivo.
        Utiliza un término de velocidad para acelerar la convergencia y almacena el historial de posiciones, normas y velocidades.
        Imprime una tabla con los resultados de cada iteración.
        
        Retorna:
        - x_historico: Lista con el historial de posiciones x en cada iteración.
        """
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
            self.data_grad_momentum.append((i+1, x0.tolist(), nomra_grad, vi.tolist()))
            
        self.imprimir_tabla_tabulate(self.data_grad_momentum, ["Iteración", "x", "Norma", "velocidad"])
        return x_historico
    
        
    def descenso_gradiente_nesterov(self):
        """
        Aplica el método de descenso de gradiente con Nesterov para minimizar la función objetivo.
        Utiliza un término de velocidad y calcula el gradiente en la posición adelantada (lookahead).
        Imprime una tabla con los resultados de cada iteración.
        
        Retorna:
        - x_historico: Lista con el historial de posiciones x en cada iteración.
        """
        f = self.f
        grad_f = self.grad_f
        x0 = self.x_0
        v0 = self.v_0
        lr = self.alpha
        eta = self.eta
        max_iters = self.iteraciones
        epsilon = self.epsilon
        x_historico = [x0]
        data_grad_nesterov = []
        
        for i in range(max_iters):
            # Lookahead: calcula el gradiente en la posición adelantada
            lookahead = x0 - eta * v0
            grad_f_i = grad_f(*lookahead)
            norm_grad = np.linalg.norm(grad_f_i)
            if norm_grad < epsilon:
                break
            vi = eta * v0 + lr * grad_f_i
            xi = x0 - vi
            x0 = xi.copy()
            v0 = vi.copy()
            x_historico.append(x0)
            data_grad_nesterov.append((i+1, x0.tolist(), norm_grad, vi.tolist()))
        
        self.imprimir_tabla_tabulate(data_grad_nesterov, ["Iteración", "x", "Norma", "velocidad"])
        return x_historico
