import numpy as np
import matplotlib.pyplot as plt


f1 = lambda x: x**2 + 2*x + 1
grad_f1 = lambda x: 2*x + 2

def desenso_gradiente_simple(f,grad_f,x0,lr,max_iters, epsilon= 0.01):
    x_historico = [x0]
    for i in range(max_iters):
        print("\n")
        f_i = f(x0)
        grad_f_i = grad_f(x0)
        nomra_grad = np.linalg.norm(grad_f_i)
        if nomra_grad < epsilon:
            break
        xi = x0 - lr * grad_f_i
        x0 = xi.copy()
        print(f"Iteracion {i+1} ----- {x0}----- | Norma {nomra_grad}")
        x_historico.append(x0)
        
    return x_historico
        


# ===================== Gráficas =====================
x_0 = np.array([10])
alpha = 0.2


f1_x0 = f1(x_0)
grad_f1_x0 = grad_f1(x_0)
print(f"f1(x_0) = {f1_x0}, grad_f1(x_0) = {grad_f1_x0}")


x = np.linspace(-10, 10, 100)
f1_x = f1(x)
plt.plot(x,f1_x, label='f1(x)')
plt.plot(x_0,f1_x0, 'ro', label='x_0')
plt.legend()
#plt.show()

x_desc = desenso_gradiente_simple(f1,grad_f1,x_0,alpha,50,0.001)