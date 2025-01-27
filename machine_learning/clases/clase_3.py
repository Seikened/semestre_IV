import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Método de descenso de gradiente con line search
def gradiente_f(f, grad_f, x0, learning_rate, max_iter, c1=1e-4, c2=0.9):
    x = x0
    history = [x0]
    for i in range(max_iter):
        grad = grad_f(x)
        # Line search con condiciones de Wolfe
        while f(x - learning_rate * grad) > f(x) - c1 * learning_rate * grad.dot(grad) or grad_f(x - learning_rate * grad).dot(grad) < c2 * grad.dot(grad):
            learning_rate *= 0.5
        x = x - learning_rate * grad
        history.append(x)
    return x, history

# Función de Rosenbrock y su gradiente
a = 1
b = 100



# Rosenbrock 
f = lambda x: (1 - x[0])**2 + b * (x[1] - x[0]**2)**2
grad_f = lambda x: np.array([
    -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2),
    2 * b * (x[1] - x[0]**2)
])



f = lambda x: ( ( x[0]-1 )**2 ) + ( ( x[1]-2 )**2 ) + ( ( x[2]-3 )**2 ) 


grad_f = lambda x: np.array([
    2*(x[0]-1),
    2*(x[1]-2),
    2*(x[2]-3)
])


# Parámetros iniciales
x0 = np.array([0.0,0.0,0.0])
learning_rate = 0.1
max_iter = 150000

# Ejecutar descenso del gradiente
x_min, history = gradiente_f(f, grad_f, x0, learning_rate, max_iter)
history = np.array(history)

# Imprimir resultados
print("Resultado")
print(f"Valor mínimo: x = {x_min}, f(x) = {f(x_min)}")

# Visualización en 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Crear la malla para la superficie
x = np.linspace(-1, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = (X - 1)**2 + (Y - 2)**2 + (4 - 3)**2  # Proyección en z fija para la visualización

# Graficar la superficie
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# Graficar la trayectoria
history_z = [f(np.array([point[0], point[1], point[2]])) for point in history]
ax.plot(history[:, 0], history[:, 1], history_z, 'o-', markersize=4, label='Trayectoria')
ax.scatter(x_min[0], x_min[1], f(x_min), color='r', s=100, label='Mínimo')

# Etiquetas y leyenda
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('f($x$)')
ax.set_title('Descenso del gradiente en 3D')
ax.legend()
plt.show()