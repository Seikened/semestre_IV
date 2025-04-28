import subprocess
import sys
try:
    from mathkat import Gradiente
    from colorstreak import log
    import numpy as np
    from PIL import Image, ImageFilter

except ImportError:

    librerias_faltantes = ['mathkat', 'colorstreak', 'numpy', 'Pillow']
    subprocess.check_call([sys.executable, "-m", "pip", "install", *librerias_faltantes]) 
    from mathkat import Gradiente 
    from colorstreak import log
    import numpy as np
    from PIL import Image, ImageFilter

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



from dataclasses import dataclass , field

@dataclass
class Imagen:
    ruta: str
    ancho: int = field(init=False)
    alto: int = field(init=False)
    imagen: any = field(init=False)

    def __post_init__(self):
        self.cargar_imagen()
        log.info(f'Imagen cargada: {self.ruta}')
        log.info(f'Tamaño de la imagen: {self.ancho}x{self.alto}')
    
    
    def cargar_imagen(self):

        self.imagen = Image.open(self.ruta)
        self.ancho, self.alto = self.imagen.size

    def mostrar(self):
        self.imagen.show()

    def cambiar_tam(self, nuevo_ancho, nuevo_alto):
        self.imagen = self.imagen.resize((nuevo_ancho, nuevo_alto))
        self.ancho, self.alto = self.imagen.size

    def guardar_img(self, nueva_ruta):
        self.imagen.save(nueva_ruta)

    def aplicar_ruido_al_pixel(self, ruido):
        
        self.imagen = self.imagen.filter(ImageFilter.GaussianBlur(ruido))





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
imagen = Imagen(ruta_img)

imagen.aplicar_ruido_al_pixel(3)
