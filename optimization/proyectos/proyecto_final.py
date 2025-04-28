import subprocess
import sys
try:
    from mathkat import Gradiente
    from colorstreak import log
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mathkat"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorstreak"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    from mathkat import Gradiente 
    from colorstreak import log
    import numpy as np

def inicializador():
    log.info('LAS LIBRERIAS "mathkat", "colorstreak"  son mias puedes consultarlo con el comando "pip show mathkat" y "pip show colorstreak"')
    respuesta = input('Quieres comprobar que son mias? (s/n): ')
    if respuesta.lower() == 's':
        subprocess.check_call([sys.executable, "-m", "pip", "show", "mathkat"])
        subprocess.check_call([sys.executable, "-m", "pip", "show", "colorstreak"])
        respuesta = input('Quieres contuinuar? (s/n): ')
        if respuesta.lower() == 's':
            log.info('Continuando...')
        else:
            log.info('Saliendo...')
            sys.exit()
    else:
        log.info('Continuando...')




# ================================================ PROYECTO FINAL ===========================================================
inicializador()


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