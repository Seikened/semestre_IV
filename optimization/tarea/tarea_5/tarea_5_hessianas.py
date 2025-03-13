import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Función 1
f1 = lambda x1, x2: x2**4 + x1**3 + 3*x1**2 + 4*x2**2 - 4*x1*x2 - 5*x2 + 8
grad_f1 = lambda x1, x2: (3*x1**2 + 6*x1 - 4*x2, 4*x2**3 + 8*x2 - 4*x1 - 5)
hessiano_f1 = lambda x1, x2: [[6*x1 + 6, -4],
                              [-4, 12*(x2**2) + 8]]

# Función 2
f2 = lambda x1, x2: 2*x1*(x2**2) + 3*np.exp(x1*x2)
grad_f2 = lambda x1, x2: (2*(x2**2) + 3*x2*np.exp(x1*x2), 4*x1*x2 + 3*x1*np.exp(x1*x2))
hessiano_f2 = lambda x1, x2: [[3*x2**2*np.exp(x1*x2), 4*x2 + 3*np.exp(x1*x2)*(1 + x1*x2)],
                              [4*x2 + 3*np.exp(x1*x2)*(1 + x1*x2), 4*x1 + 3*(x1**2)*np.exp(x1*x2)]]

# Función 3
f3 = lambda x1, x2, x3: x1**2 + x2**2 + 2*x3**2
grad_f3 = lambda x1, x2, x3: (2*x1, 2*x2, 4*x3)
hessiano_f3 = lambda x1, x2, x3: [[2, 0, 0],
                                  [0, 2, 0],
                                  [0, 0, 4]]

# Función 4
f4 = lambda x1, x2: np.log(x1**2 + 2*x1*x2 + 3*x2**2)
grad_f4 = lambda x1, x2: ((2*x1 + 2*x2)/(x1**2 + 2*x1*x2 + 3*x2**2),
                          (2*x1 + 6*x2)/(x1**2 + 2*x1*x2 + 3*x2**2))
hessiano_f4 = lambda x1, x2: [
    [(-2*x1**2 - 4*x1*x2 + 2*x2**2)/(x1**2 + 2*x1*x2 + 3*x2**2)**2,
     (-2*x1**2 - 12*x1*x2 - 6*x2**2)/(x1**2 + 2*x1*x2 + 3*x2**2)**2],
    [(-2*x1**2 - 12*x1*x2 - 6*x2**2)/(x1**2 + 2*x1*x2 + 3*x2**2)**2,
     (2*x1**2 - 12*x1*x2 - 18*x2**2)/(x1**2 + 2*x1*x2 + 3*x2**2)**2]
]



# ===================== Gráficas =====================
# ========== Graficar funciones de 2 variables con su gradiente ==========

def plot_function_and_gradient_2d(f, grad_f, title, x1_range=(-3,3), x2_range=(-3,3), points=40):
    """
    Crea una figura con 2 subplots:
    - Subplot 1 (3D): Superficie z = f(x1, x2)
    - Subplot 2 (2D): Contorno de f + campo de gradientes
        Hecha por CHATGPT
    """
    fig = plt.figure(figsize=(12, 5))
    
    # --- Subplot 1: Gráfica 3D ---
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    x1_vals = np.linspace(x1_range[0], x1_range[1], points)
    x2_vals = np.linspace(x2_range[0], x2_range[1], points)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f(X1, X2)
    
    ax3d.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax3d.set_xlabel('x1')
    ax3d.set_ylabel('x2')
    ax3d.set_zlabel('f(x1, x2)')
    ax3d.set_title(f'{title} - Superficie 3D')
    
    # --- Subplot 2: Contorno 2D + gradiente ---
    ax2d = fig.add_subplot(1, 2, 2)
    cs = ax2d.contour(X1, X2, Z, levels=20, cmap='viridis')
    ax2d.clabel(cs, inline=True, fontsize=8)
    
    step = points // 8 if points > 8 else 1
    X1_grad = X1[::step, ::step]
    X2_grad = X2[::step, ::step]
    GradX1 = np.zeros_like(X1_grad)
    GradX2 = np.zeros_like(X2_grad)
    
    rows, cols = X1_grad.shape
    for i in range(rows):
        for j in range(cols):
            gx1, gx2 = grad_f(X1_grad[i, j], X2_grad[i, j])
            GradX1[i, j] = gx1
            GradX2[i, j] = gx2
    
    ax2d.quiver(X1_grad, X2_grad, GradX1, GradX2, color='r', alpha=0.8)
    ax2d.set_xlabel('x1')
    ax2d.set_ylabel('x2')
    ax2d.set_title(f'{title} - Contorno + Gradiente')
    
    plt.tight_layout()
    plt.show()

# ========== Graficar función de 3 variables (f3) “slicers” de x3 ==========

def plot_function_3var_slices(f_3d, title, x3_values=[-2, -1, 0, 1, 2],
                              x1_range=(-5,5), x2_range=(-5,5), points=50):
    """
    Grafica la función f_3d(x1, x2, x3) en 3D para varios valores fijos de x3.
    Cada “slice” se dibuja como una superficie:
        z = f_3d(x1, x2, x3 fijo)
        Hecha por CHATGPT
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    x1_vals = np.linspace(x1_range[0], x1_range[1], points)
    x2_vals = np.linspace(x2_range[0], x2_range[1], points)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    for x3 in x3_values:
        Z = f_3d(X1, X2, x3)
        ax.plot_surface(X1, X2, Z, alpha=0.5, label=f'x3={x3}')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2, x3)')
    ax.set_title(title)
    plt.legend()
    plt.show()

# ========== Graficación ==========
plot_function_and_gradient_2d(f1, grad_f1, 'Función 1')
plot_function_and_gradient_2d(f2, grad_f2, 'Función 2')
plot_function_3var_slices(f3, 'Función 3 (Slices en x3)', x3_values=[-2, -1, 0, 1, 2])
plot_function_and_gradient_2d(f4, grad_f4, 'Función 4')

