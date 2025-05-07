from mathkat import Gradiente 
from colorstreak import log
import numpy as np
import cv2
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from numba import jit


@jit(nopython=True)
def psnr(img1, img2, max_pixel: float = 255.0):
    """Calcula PSNR (dB) entre dos imágenes."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_pixel / np.sqrt(mse))




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
        ruido = np.random.normal(mean, sigma, self.imagen.shape)
        noisy = self.imagen + ruido
        #noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        self.imagen = noisy

    def img_vector(self):
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
    def _laplaciano(self, u_vec: np.ndarray):
        """Laplaciano discreto de 5 puntos sobre la imagen 2‑D."""
        u = u_vec.reshape(self.H, self.W)
        lap = (
            -4.0 * u +
            np.roll(u,  1, 0) + np.roll(u, -1, 0) +
            np.roll(u,  1, 1) + np.roll(u, -1, 1)
        )
        return lap.flatten()

    # ---------- API requerida por Gradiente ----------
    def func(self, u_vec: np.ndarray):
        """Devuelve J(u)."""
        diff = u_vec - self.f_vec
        lap  = self._laplaciano(u_vec)
        return 0.5 * np.dot(diff, diff) + 0.5 * self.lam * np.dot(lap, lap)

    def grad(self, u_vec: np.ndarray):
        """Devuelve ∇J(u)."""
        #   ∇J(u) = (u - f) - λ Δu   (signo menos: ecuaciones de Euler‑Lagrange)
        return (u_vec - self.f_vec) - self.lam * self._laplaciano(u_vec)


# ================================================ PROYECTO FINAL ===========================================================


def mensaje(psnr, ssim, min, iter,):
    return (f"Nesterov\nPSNR: {psnr:.1f} dB | SSIM: {ssim:.3f}"
            f"\nMin: {min:.2e} | Iter: {iter}")
    
# ==================== Ejemplo de uso (actualizado) ====================

try:
    ruta_base = '/Users/ferleon/Documents/GitHub/semestre_IV/optimization/proyectos/'
except:
    raise FileNotFoundError("No se pudo encontrar la ruta base. Asegúrate de que la ruta sea correcta. o configura la tuya")
ruta_img  = ruta_base + 'men_moon.jpg'   

imagen_original = Imagen(ruta_img)

reductor = 10

imagen_original.cambiar_tam(imagen_original.ancho // reductor, imagen_original.alto // reductor)
img_original = imagen_original.imagen.copy()
f_img = imagen_original.imagen.astype(np.float32)

imagen_ruido = Imagen(ruta_img)
imagen_ruido.cambiar_tam(imagen_ruido.ancho // reductor, imagen_ruido.alto // reductor)


# ================================== Ruido y restauración ==========================================================
for ruido in [0, 10, 20, 30, 40, 50]:
    
    imagen_ruido.aplicar_ruido_al_pixel(ruido)         
    f_noisy = imagen_ruido.imagen.astype(np.float32)

    energia = EnergiaL2(f_noisy, lam=0.2)
    alphas = [0.001, 0.02, 0.5, 1.0, 2.0]

    gradientes = []
    for alpha in alphas:
        grad_u = Gradiente(
            f       = energia.func,
            grad_f  = energia.grad,
            x_0     = energia.f_vec.copy(),
            v_0     = np.zeros_like(energia.f_vec),
            alpha   = alpha,                                   
            iteraciones = 1500,
            epsilon = 1e-6,
            eta     = 0.8
        )
        gradientes.append(grad_u)

    for gradiente in gradientes:

        metodo = 'nesterov' 
        grad_nesterov = gradiente.nesterov()
        min_nesterov = gradiente.mas_chico
        iter_nesterov = gradiente.iteracion_mas_chico
        foto_nesterov = gradiente.x_historico[-1].reshape(energia.H, energia.W)


        metodo = 'momentum'
        grad_momentum = gradiente.momentum()
        min_momentum = gradiente.mas_chico
        iter_momentum = gradiente.iteracion_mas_chico
        foto_momentum = gradiente.x_historico[-1].reshape(energia.H, energia.W)


        metodo = 'simple'
        grad_simple = gradiente.simple()
        min_simple = gradiente.mas_chico
        iter_simple = gradiente.iteracion_mas_chico
        foto_simple = gradiente.x_historico[-1].reshape(energia.H, energia.W)



# ========================================== Visualización de imágenes ==========================================
        fig, axs = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f'Ruido: {ruido} | Alpha: {gradiente.alpha} | Beta: {gradiente.eta}', fontsize=16)

        # Leer imágenes restauradas
        img_ruido = imagen_ruido.imagen
        img_simple = foto_simple
        img_momentum = foto_momentum
        img_nesterov = foto_nesterov

        # ---- Métricas de calidad ----
        psnr_simple   = psnr(img_original, img_simple)
        ssim_simple   = ssim(img_original.astype(np.uint8),
                            img_simple.astype(np.uint8),
                            data_range=255)

        psnr_momentum = psnr(img_original, img_momentum)
        ssim_momentum = ssim(img_original.astype(np.uint8),
                            img_momentum.astype(np.uint8),
                            data_range=255)

        psnr_nesterov = psnr(img_original, img_nesterov)
        ssim_nesterov = ssim(img_original.astype(np.uint8),
                            img_nesterov.astype(np.uint8),
                            data_range=255)

        # Fila 1: Imagen original
        for j, metodo in enumerate(['simple', 'momentum', 'nesterov']):
            axs[0, j].imshow(img_original, cmap='gray')
            axs[0, j].set_title('Original')
            axs[0, j].axis('off')

        # Fila 2: Imagen con ruido
        for j, metodo in enumerate(['simple', 'momentum', 'nesterov']):
            axs[1, j].imshow(img_ruido, cmap='gray')
            axs[1, j].set_title(f'Ruido')
            axs[1, j].axis('off')

        # Sección de imágenes restauradas
        # Simple
        axs[2, 0].imshow(img_simple, cmap='gray')
        axs[2, 0].set_title(mensaje(
            psnr_simple, ssim_simple, min_simple, iter_simple
        ))
        # Momentum
        axs[2, 0].axis('off')
        axs[2, 1].imshow(img_momentum, cmap='gray')
        axs[2, 1].set_title(mensaje(
            psnr_momentum, ssim_momentum, min_momentum, iter_momentum
        ))
        
        # Nesterov
        axs[2, 1].axis('off')
        axs[2, 2].imshow(img_nesterov, cmap='gray')
        axs[2, 2].set_title(mensaje(
            psnr_nesterov, ssim_nesterov, min_nesterov, iter_nesterov
        ))
        axs[2, 2].axis('off')


        plt.tight_layout()
        try:
            plt.savefig(ruta_base + f'ruido_{ruido}/' + f'comparacion_{ruido}_{gradiente.alpha}_{gradiente.eta}.png')
        except FileNotFoundError:
            import os
            os.makedirs(ruta_base + f'ruido_{ruido}/', exist_ok=True)
            plt.savefig(ruta_base + f'ruido_{ruido}/' + f'comparacion_{ruido}_{gradiente.alpha}_{gradiente.eta}.png')
        except Exception as e:
            print(f"Error al guardar la imagen: {e}")
        
        plt.close()