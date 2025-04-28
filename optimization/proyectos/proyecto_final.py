#from mathkat import Gradiente 
from colorstreak import log
import numpy as np
import cv2
from rich.table import Table
from rich.console import Console
import functools
from dataclasses import dataclass, field


# ================================================ Gradiente temporar (borrar despues de las pruebas) ===========================================================



console = Console()

def imprimir_tabla(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        resultado = func(self, *args, **kwargs)
        if hasattr(self, 'data'):
            nombre_func = func.__name__

            match nombre_func:
                case "simple":
                    headers = ["Iteración", "x", "Norma"]
                case "momentum":
                    headers = ["Iteración", "x", "Norma", "velocidad"]
                case "nesterov":
                    headers = ["Iteración", "x", "Norma", "velocidad"]
            table = Table(title=f"[bold magenta]Resultados de {nombre_func.capitalize()}[/bold magenta]", show_lines=True)
            for header in headers:
                table.add_column(header.upper(), justify="center", style="yellow", no_wrap=True)
            for row in self.data:
                table.add_row(*[str(x) for x in row])
            console.print(table)
        return resultado
    return wrapper

@dataclass
class Gradiente:
    """
    Clase que representa una función objetivo y sus métodos de optimización mediante descenso de gradiente.
    Permite aplicar diferentes variantes del descenso de gradiente y visualizar los resultados en tablas.
    """

    f : callable
    grad_f : callable
    x_0 : np.ndarray
    v_0 : np.ndarray
    alpha : float
    iteraciones : int
    epsilon : float
    eta : float
    x_historico : list = field(default_factory=list, init=False)
    data : list = field(default_factory=list, init=False)

    @staticmethod
    def _desempaquetar(func, x_0):
        try:
            return func(*x_0)
        except TypeError:
            return func(x_0)

        

    def reset(self):
        """
        Reinicia el historial de posiciones y datos.
        """
        self.x_historico = []
        self.data = []
    

    #@imprimir_tabla
    def simple(self):
        """
        Realiza el descenso de gradiente estándar para minimizar la función objetivo.
        En cada iteración, actualiza la posición usando el gradiente y almacena el historial de posiciones y normas.
        Imprime una tabla con los resultados de cada iteración.
        
        Retorna:
        - x_historico: Lista con el historial de posiciones x en cada iteración.
        """
        self.reset()
        f = self.f
        grad_f = self.grad_f
        x0 = self.x_0
        lr = self.alpha
        max_iters = self.iteraciones
        epsilon = self.epsilon
        self.x_historico = [x0]
        
        for i in range(max_iters):
            f_i = self._desempaquetar(f, x0)
            grad_f_i = self._desempaquetar(grad_f, x0)
            norm_grad = np.linalg.norm(grad_f_i)
            if norm_grad < epsilon: 
                break
            xi = x0 - lr * grad_f_i
            x0 = xi.copy()
            log.info(f"Iteración {i+1}: x0 = {x0}, grad_f = {grad_f_i}, norma_grad = {norm_grad}")
            self.x_historico.append(x0)
            self.data.append((i+1, x0.tolist(), norm_grad))
        return self.x_historico


    #@imprimir_tabla
    def momentum(self):
        """
        Aplica el método de descenso de gradiente con momentum para minimizar la función objetivo.
        Utiliza un término de velocidad para acelerar la convergencia y almacena el historial de posiciones, normas y velocidades.
        Imprime una tabla con los resultados de cada iteración.
        
        Retorna:
        - x_historico: Lista con el historial de posiciones x en cada iteración.
        """
        self.reset()
        f = self.f
        grad_f = self.grad_f
        x0 = self.x_0
        v0 = self.v_0
        lr = self.alpha
        eta = self.eta
        max_iters = self.iteraciones
        epsilon = self.epsilon
        self.x_historico = [x0]
        
        for i in range(max_iters):
            f_i = self._desempaquetar(f, x0)
            grad_f_i = self._desempaquetar(grad_f, x0)
            norma_grad = np.linalg.norm(grad_f_i)
            if norma_grad < epsilon:
                break
            vi = eta * v0 + lr * grad_f_i
            xi = x0 - vi
            x0 = xi.copy()
            v0 = vi.copy()
            self.x_historico.append(x0)
            self.data.append((i+1, x0.tolist(), norma_grad, vi.tolist()))
            log.info(f"Iteración {i+1}: x0 = {x0}, grad_f = {grad_f_i}, norma_grad = {norma_grad}, velocidad = {vi}")
        return self.x_historico
    
    
    #@imprimir_tabla
    def nesterov(self):
        """
        Aplica el método de descenso de gradiente con Nesterov para minimizar la función objetivo.
        Utiliza un término de velocidad y calcula el gradiente en la posición adelantada (lookahead).
        Imprime una tabla con los resultados de cada iteración.
        
        Retorna:
        - x_historico: Lista con el historial de posiciones x en cada iteración.
        """
        self.reset()
        f = self.f
        grad_f = self.grad_f
        x0 = self.x_0
        v0 = self.v_0
        lr = self.alpha
        eta = self.eta
        max_iters = self.iteraciones
        epsilon = self.epsilon
        self.x_historico = [x0]
        chico = []
        
        for i in range(max_iters):
            lookahead = x0 - eta * v0
            grad_f_i = self._desempaquetar(grad_f, lookahead)
            norm_grad = np.linalg.norm(grad_f_i)
            if norm_grad < epsilon:
                break
            vi = eta * v0 + lr * grad_f_i
            xi = x0 - vi
            x0 = xi.copy()
            v0 = vi.copy()
            self.x_historico.append(x0)
            self.data.append((i+1, x0.tolist(), norm_grad, vi.tolist()))
            chico.append(norm_grad)
            log.info(f"Iteración {i+1}: x0 = {x0}, grad_f = {grad_f_i}, norma_grad = {norm_grad}")
        mas_chico = min(chico)
        iteracion_mas_chico = chico.index(mas_chico) + 1
        log.info(f"Iteración con menor norma del gradiente: en la iteración: {iteracion_mas_chico} con valor {mas_chico}")
        
        return self.x_historico



# =============================================== Imagen ===========================================================
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
        ruido = np.random.normal(mean, sigma, self.imagen.shape).astype(np.int16)
        noisy = self.imagen.astype(np.int16) + ruido
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        self.imagen = noisy

    def img_vector(self) -> np.ndarray:
        return self.imagen.flatten()

# =============================================== Energía L2 (λ‖∇u‖²) ===========================================================
class EnergiaL2:
    """
    Encapsula la energía

        J(u) = 0.5‖u - f‖² + 0.5·λ‖∇u‖²

    y su gradiente discreto usando el laplaciano de cinco puntos.
    Trabaja con vectores planos (`u_vec`) para ser compatible con la
    clase Gradiente y realiza el reshape internamente.
    """
    def __init__(self, f_img: np.ndarray, lam: float = 0.2):
        self.f_img = f_img.astype(np.float32)
        self.lam   = float(lam)
        self.H, self.W = self.f_img.shape
        self.f_vec = self.f_img.flatten()

    # ---------- helpers ----------
    def _laplaciano(self, u_vec: np.ndarray) -> np.ndarray:
        """Laplaciano discreto de 5 puntos sobre la imagen 2‑D."""
        u = u_vec.reshape(self.H, self.W)
        lap = (
            -4.0 * u +
            np.roll(u,  1, 0) + np.roll(u, -1, 0) +
            np.roll(u,  1, 1) + np.roll(u, -1, 1)
        )
        return lap.flatten()

    # ---------- API requerida por Gradiente ----------
    def func(self, u_vec: np.ndarray) -> float:
        """Devuelve J(u)."""
        diff = u_vec - self.f_vec
        lap  = self._laplaciano(u_vec)
        return 0.5 * np.dot(diff, diff) + 0.5 * self.lam * np.dot(lap, lap)

    def grad(self, u_vec: np.ndarray) -> np.ndarray:
        """Devuelve ∇J(u)."""
        #   ∇J(u) = (u - f) - λ Δu   (signo menos: ecuaciones de Euler‑Lagrange)
        return (u_vec - self.f_vec) - self.lam * self._laplaciano(u_vec)


# ================================================ PROYECTO FINAL ===========================================================


# ==================== Ejemplo de uso (actualizado) ====================

ruta_base = '/Users/ferleon/Documents/GitHub/semestre_IV/optimization/proyectos/'
ruta_img  = ruta_base + 'men_moon.jpg'   # ajusta a la imagen que quieras

# --- 1. Cargar la imagen y generar versión ruidosa ---
imagen_original = Imagen(ruta_img)
log.info(f"Imagen original: {imagen_original.ancho}x{imagen_original.alto}")

# Redimensionar para pruebas rápidas
imagen_original.cambiar_tam(imagen_original.ancho // 4, imagen_original.alto // 4)
f_img = imagen_original.imagen.astype(np.float32)

# Crear copia ruidosa para el experimento
imagen_ruido = Imagen(ruta_img)
imagen_ruido.cambiar_tam(imagen_ruido.ancho // 4, imagen_ruido.alto // 4)
imagen_ruido.aplicar_ruido_al_pixel(25)               # σ = 25
f_noisy = imagen_ruido.imagen.astype(np.float32)
cv2.imwrite(ruta_base + 'imagen_ruido.png', f_noisy)

# --- 2. Construir problema de energía L2 ---
energia = EnergiaL2(f_noisy, lam=0.2)                 # λ = 0.2 (ajusta a gusto)

# --- 3. Ejecutar optimizador ---
opt = Gradiente(
    f       = energia.func,
    grad_f  = energia.grad,
    x_0     = energia.f_vec.copy(),                   # inicialización = imagen ruidosa
    v_0     = np.zeros_like(energia.f_vec),
    #alpha   = 5e-4,
    alpha   = 3e-2,                                   
    iteraciones = 1500,
    epsilon = 1e-6,
    eta     = 0.8
)

opt.nesterov()                                        # también .momentum() o .simple()
u_final = opt.x_historico[-1].reshape(energia.H, energia.W).astype(np.uint8)
cv2.imwrite(ruta_base + 'imagen_denoise.png', u_final)