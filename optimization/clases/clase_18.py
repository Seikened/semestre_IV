import numpy as np
import matplotlib.pyplot as plt




def nomra(*args):
    """
    Función que calcula la norma de un vector manualmente
    """
    return np.sqrt(np.sum(np.fromiter((np.pow(x,2) for x in args), dtype=float)))


def desenso_gradiente_simple(f,grad_f,x0,lr,max_iters, epsilon= 0.01):
    x_historico = [x0]
    for i in range(max_iters):
        print("\n")
        f_i = f(*x0)
        grad_f_i = grad_f(*x0)
        nomra_grad = np.linalg.norm(grad_f_i)
        if nomra_grad < epsilon: # Criterio de paro 
            break
        xi = x0 - lr * grad_f_i
        x0 = xi.copy()
        print(f"Iteracion {i+1} ----- {x0}----- | Norma {nomra_grad}")
        x_historico.append(x0)
        
    return x_historico


def desenso_gradiente_momentum(f,grad_f,x0,v0,lr,max_iters,eta= 0.9, epsilon= 0.01):
    x_historico = [x0]
    for i in range(max_iters):
        print("\n")
        f_i = f(*x0)
        grad_f_i = grad_f(*x0)
        nomra_grad = np.linalg.norm(grad_f_i)
        if nomra_grad < epsilon: # Criterio de paro 
            break
        vi = eta * v0 + lr * grad_f_i
        xi = x0 - vi
        x0 = xi.copy()
        v0 = vi.copy()
        print(f"Iteracion {i+1} ----- {x0}----- | Norma {nomra_grad} | Velocidad {vi}")
        x_historico.append(x0)
        
    return x_historico
        
# ===================== Funciones =====================


f1 = lambda x: x**2 + 2*x + 1
grad_f1 = lambda x: np.array([2*x + 2])

f2 = lambda x1,x2: x1**2 + (2*x2**2)
grad_f2 = lambda x1,x2: np.array([2*x1, 4*x2])

f3 = lambda x1,x2,x3: x1**2 + x2**2 + 2*x3**2
grad_f3 = lambda x1,x2,x3: np.array([2*x1, 2*x2, 4*x3])



# ===================== Gráficas =====================
x_0 = np.array([10])
alpha = 0.2


f1_x0 = f1(x_0)
grad_f1_x0 = grad_f1(x_0)
print(f"f1(x_0) = {f1_x0}, grad_f1(x_0) = {grad_f1_x0}")


x = np.linspace(-10, 10, 100)
f1_x = f1(x)
f2_x = f2(x,x)
# plt.plot(x,f1_x, label='f1(x)')
# plt.plot(x_0,f1_x0, 'ro', label='x_0')
# plt.plot(x,grad_f1(x), label='grad_f1(x)')
# plt.plot(x, f2(x,x), label='f2(x1,x2)')
# plt.plot(x, grad_f2(x,x)[0], label='grad_f2(x1,x2)')
# plt.legend()
#plt.show()



x_0 = np.array([-5,-2])
alpha = 0.1
#x_f2 = desenso_gradiente_simple(f2,grad_f2,x_0,alpha,50,0.001)

x_0 = np.array([-1,-1,-1])
alpha = 0.1
#x_f3 = desenso_gradiente_simple(f3,grad_f3,x_0,alpha,50,0.001)


# ================= Función 1 =================

x_0 = np.array([10])
alpha = 0.1
v_0 = np.array([0])
iteraciones = 50
epsilon = 0.001
eta = 0.9

print("FUNCION SIN MOMENTUM")
x_f1 = desenso_gradiente_simple(f1,grad_f1,x_0,alpha,iteraciones,epsilon)
print("Momentum F1")
momentum = desenso_gradiente_momentum(f1,grad_f1,x_0,v_0,alpha,iteraciones,eta,epsilon)

# ================= Función 2 =================

x_0 = np.array([-5,-2])
alpha = 0.1
v_0 = np.array([0,0])
iteraciones = 100
epsilon = 0.001
eta = 0.9

print("FUNCION SIN MOMENTUM")
x_f1 = desenso_gradiente_simple(f2,grad_f2,x_0,alpha,iteraciones,epsilon)
print("Momentum F1")
momentum = desenso_gradiente_momentum(f2,grad_f2,x_0,v_0,alpha,iteraciones,eta,epsilon)

# ================= Función 3 =================
x_0 = np.array([-1,-1,-1])
alpha = 0.1
v_0 = np.array([0,0,0])
iteraciones = 100
epsilon = 0.001
eta = 0.9
print("FUNCION SIN MOMENTUM")
x_f1 = desenso_gradiente_simple(f3,grad_f3,x_0,alpha,iteraciones,epsilon)
print("Momentum F1")
momentum = desenso_gradiente_momentum(f3,grad_f3,x_0,v_0,alpha,iteraciones,eta,epsilon)