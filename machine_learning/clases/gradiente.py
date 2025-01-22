import numpy as np
import matplotlib.pyplot as plt


def  gradiente(f,grad_f,x0,learning_rate,max_iter):
    x = x0
    for i in range(max_iter):
        grad = grad_f(x)
        x = x - learning_rate * grad
        print(f"Iteración {i+1}: x = {x}, f(x) = {f(x)}")
    return x


import numpy as np

def  gradiente_f(f,grad_f,x0,learning_rate,max_iter,c1=1e-4, c2 = 0.9):
    x = x0
    history = [x0]
    for i in range(max_iter):
        grad = grad_f(x)
        while f(x - learning_rate * grad) > f(x) -c1 * learning_rate * grad.dot(grad) or grad_f(x - learning_rate * grad).dot(grad) < c2 * grad.dot(grad):
            learning_rate *= 0.5
        x = x -learning_rate * grad
        history.append(x)
    return x,history


f = lambda x: x[0]**2 + x[1]**2
graf_f = lambda x: np.array( [2*x[0],2*x[1]] )
x0 = np.array( [1.0 , 1.0] )
learning_rate = 0.1
max_iter = 100

x_min = gradiente(f,graf_f,x0,learning_rate,max_iter)
print(f"Valor minimo: x = {x_min}, f(x) = {f(x_min)}")



f = lambda x: x[0]**2 + x[1]**2
graf_f = lambda x: np.array( [2*x[0],2*x[1]] )
x0 = np.array( [1.0 , 1.0] )
learning_rate = 0.1
max_iter = 100

x_min = gradiente(f,graf_f,x0,learning_rate,max_iter)
#print(f"Valor minimo: x = {x_min}, f(x) = {f(x_min)}")




f = lambda x: ((x[0] -2)**2) + ((x[0] +3)**2)
graf_f = lambda x: np.array( 2 * (x[0]) - 4, 2 * (x[0]) + 6  )
x0 = np.array( [0.0 , 0.0] )
learning_rate = 0.1
max_iter = 100


x_min,history = gradiente_f(f,graf_f,x0,learning_rate,max_iter,c1=1e-4, c2 = 0.9)
history = np.array(history)
print("Resultado")
print(f"Valor minimo: x = {x_min}, f(x) = {f(x_min)}")


x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x,y)
Z = X**2 + Y**2


# Graficar
plt.figure(figsize=(10,6))
plt.contour(X,Y,Z, levels=50, cmap='viridis')
plt.plot(history[:,0],history[:,1], 'o-',markersize=4, label='Trayectoria')
plt.plot(x_min[0],x_min[1], 'rx',markersize=10, label='Minimo')
plt.xlabel('$x_0$')
plt.ylabel('$x_1$')
plt.title('Desenso del gradiente')
plt.legend()
plt.grid(True)
plt.show()


