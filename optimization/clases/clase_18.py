from mathkat import Funcion
import numpy as np



# ===================== Funciones =====================

# === FUNCIONES DE LA CLASE ===
# Funciones 1 de la clase
f1 = lambda x: x**2 + 2*x + 1
grad_f1 = lambda x: np.array([2*x + 2])

f2 = lambda x1,x2: x1**2 + (2*x2**2)
grad_f2 = lambda x1,x2: np.array([2*x1, 4*x2])

f3 = lambda x1,x2,x3: x1**2 + x2**2 + 2*x3**2
grad_f3 = lambda x1,x2,x3: np.array([2*x1, 2*x2, 4*x3])


# === FUNCIONES DE TAREA ===
# Función 1 de tarea
f1_t = lambda x1, x2: x2**4 + x1**3 + 3*x1**2 + 4*x2**2 - 4*x1*x2 - 5*x2 + 8
grad_f1_t = lambda x1, x2: np.array([3*x1**2 + 6*x1 - 4*x2, 4*x2**3 + 8*x2 - 4*x1 - 5])


# Función 2 de tarea
f2_t = lambda x1, x2: 2*x1*(x2**2) + 3*np.exp(x1*x2)
grad_f2_t = lambda x1, x2: np.array([2*(x2**2) + 3*x2*np.exp(x1*x2), 4*x1*x2 + 3*x1*np.exp(x1*x2)])

# Función 3 de tarea
f3_t = lambda x1, x2, x3: x1**2 + x2**2 + 2*x3**2
grad_f3_t = lambda x1, x2, x3: np.array([2*x1, 2*x2, 4*x3])

# Función 4 de tarea
f4_t = lambda x1, x2: np.log(x1**2 + 2*x1*x2 + 3*x2**2)
grad_f4_t = lambda x1, x2: np.array([ (2*x1 + 2*x2)/(x1**2 + 2*x1*x2 + 3*x2**2),
                          (2*x1 + 6*x2)/(x1**2 + 2*x1*x2 + 3*x2**2)])




# ================= Función 1 de clase =================

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

# ================= Función 2 de clase =================

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

# ================= Función 3 de clase =================
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


# ================= Función 1 de tarea =================
x_0 = np.array([-1,-2])
alpha = 0.1
v_0 = np.array([0,0])
iteraciones = 50
epsilon = 0.001
eta = 0.9
f1_t = Funcion(f1_t,grad_f1_t,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION DE TAREA SIN MOMENTUM")
f1_t.desenso_gradiente_simple()
print("Momentum F1 Tarea")
f1_t.desenso_gradiente_momentum()
# ================= Función 2 de tarea =================
x_0 = np.array([-1,-2])
alpha = 0.1
v_0 = np.array([0,0])
iteraciones = 50
epsilon = 0.001
eta = 0.9
f2_t = Funcion(f2_t,grad_f2_t,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION DE TAREA SIN MOMENTUM")
f2_t.desenso_gradiente_simple()
print("Momentum F2 Tarea")
f2_t.desenso_gradiente_momentum()
# ================= Función 3 de tarea =================
x_0 = np.array([-1,-2,-3])
alpha = 0.1
v_0 = np.array([0,0,0])
iteraciones = 50
epsilon = 0.001
eta = 0.9
f3_t = Funcion(f3_t,grad_f3_t,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION DE TAREA SIN MOMENTUM")
f3_t.desenso_gradiente_simple()
print("Momentum F3 Tarea")
f3_t.desenso_gradiente_momentum()
# ================= Función 4 de tarea =================
x_0 = np.array([-1,-2])
alpha = 0.1
v_0 = np.array([0,0])
iteraciones = 50
epsilon = 0.001
eta = 0.9
f4_t = Funcion(f4_t,grad_f4_t,x_0,v_0,alpha,iteraciones,epsilon,eta)
print("FUNCION DE TAREA SIN MOMENTUM")
f4_t.desenso_gradiente_simple()
print("Momentum F4 Tarea")
f4_t.desenso_gradiente_momentum()
print("FIN DE LA EJECUCIÓN")