import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# Modo funcional
# ==================================================
x = np.linspace(1,10,100)
y = np.sin(x)
y2 = np.cos(x)

# Recibe una lista de valores en x y en y
plt.plot(x, y)
plt.plot(x, y2)
plt
#plt.show()
plt.scatter(x, y, color="lightgreen", label="sin(x)")
# Abajo
plt.legend(loc="lower left")
plt.xlabel("x")
plt.ylabel("sin(x)", rotation=0, labelpad=5)
plt.title("Ejemplo 1 de grafica")
plt.grid(axis="y")
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(-1, 1.1, 0.1))
plt.show()