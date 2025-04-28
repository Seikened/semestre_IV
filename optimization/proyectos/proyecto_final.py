from mathkat import Gradiente 
from colorstreak import log
import numpy as np
import cv2
from dataclasses import dataclass , field

@dataclass
class Imagen:
    ruta: str
    ancho: int = field(init=False)
    alto: int = field(init=False)
    imagen: any = field(init=False)

    def __post_init__(self):
        self.cargar_imagen()
    
    
    def cargar_imagen(self):
        self.imagen = cv2.imread(self.ruta, cv2.IMREAD_GRAYSCALE)
        if self.imagen is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en {self.ruta}")
        self.alto, self.ancho = self.imagen.shape

    def mostrar(self):
        cv2.imshow('Imagen', self.imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def cambiar_tam(self, nuevo_ancho, nuevo_alto):
        self.imagen = cv2.resize(self.imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
        self.ancho, self.alto = nuevo_ancho, nuevo_alto

    def guardar_img(self, nueva_ruta):
        cv2.imwrite(nueva_ruta, self.imagen)

    def aplicar_ruido_al_pixel(self, sigma: float = 60.0, mean: float = 0.0):
        """
        Agrega ruido gaussiano aditivo controlando media y desviación estándar.
        El resultado se satura al rango [0, 255] y se almacena de nuevo en self.imagen.

        Parámetros
        ----------
        sigma : float
            Desviación estándar del ruido (cuanto mayor, más intenso).
        mean : float
            Media del ruido; normalmente 0.
        """
        # Trabajamos en int16 para evitar overflow en la suma
        ruido = np.random.normal(mean, sigma, self.imagen.shape).astype(np.int16)
        noisy = self.imagen.astype(np.int16) + ruido
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        self.imagen = noisy





# ================================================ PROYECTO FINAL ===========================================================
#inicializador()


# # Ejemplo de uso
# f = lambda x, y: (x-1)**2 + (y-2)**2
# grad_f = lambda x, y: np.array([2*(x-1), 2*(y-2)])
# x_0 = np.array([0.0, 0.0])
# v_0 = np.array([0.0, 0.0])
# alpha = 0.1
# iteraciones = 10
# epsilon = 1e-6
# eta = 0.9

# funcion = Gradiente(f, grad_f, x_0, v_0, alpha, iteraciones, epsilon, eta)
# funcion.simple()
# funcion.momentum()
# funcion.nesterov()

funcion_objetivo = lambda u,f,λ,u_grad: ( 0.5 * (abs(u - f)**2) ) + ((λ/2) * (abs(u_grad)**2) )



# # Ejemplo de uso
ruta_base = '/Users/ferleon/Documents/GitHub/semestre_IV/optimization/proyectos/'
ruta_img = ruta_base + 'fer.jpeg'
ruta_img = ruta_base + 'F.png'
ruta_img = ruta_base + 'men_moon.jpg'

imagen = Imagen(ruta_img)
imagen.mostrar()

imagen.aplicar_ruido_al_pixel(25)
imagen.mostrar()
