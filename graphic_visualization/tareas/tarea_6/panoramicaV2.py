import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json



alto,ancho = 800,600
ruta =  os.path.join(os.path.dirname(__file__), "img")
img1 = cv2.imread(os.path.join(ruta, "img1.jpeg"))
img2 = cv2.imread(os.path.join(ruta, "img2.jpeg"))
img3 = cv2.imread(os.path.join(ruta, "img3.jpeg"))
img4 = cv2.imread(os.path.join(ruta, "img4.jpeg"))
img5 = cv2.imread(os.path.join(ruta, "img5.jpeg"))

lista_natural_imgs = []
lista_gres_imgs = []
for img in [img1, img2, img3, img4, img5]:
    img = cv2.resize(img, (ancho, alto))
    img1_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lista_natural_imgs.append(img)
    lista_gres_imgs.append(img1_gray)


# =============================================================================
# FUNCIIONES
# =============================================================================

def probar_stitcher_con_subconjunto(imagenes, indices):
    subset = [imagenes[i] for i in indices]
    print(f"Probando con imágenes: {indices}")
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, resultado = stitcher.stitch(subset)
    if status != cv2.Stitcher_OK:
        print("   No se pudo unir este subconjunto.")
    else:
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
        plt.title(f"Panorámica con imágenes: {indices}")
        plt.axis("off")
        plt.show()


def test_one_by_one():
    probar_stitcher_con_subconjunto(lista_natural_imgs, [0, 1])
    probar_stitcher_con_subconjunto(lista_natural_imgs, [1, 2])
    probar_stitcher_con_subconjunto(lista_natural_imgs, [2, 3])
    probar_stitcher_con_subconjunto(lista_natural_imgs, [3, 4])


def unir_imagenes_modo_facil(imgs):
    """ Funciona para unir imagenes de manera facil """
    imagenes = imgs.copy()
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, resultado = stitcher.stitch(imagenes)
    if status != cv2.Stitcher_OK:
        print("No se pudo unir las imagenes")
        return None
    return resultado



def homography(source_points,destination_points):
    """
    Calcula la matriz de homografía a partir de los puntos de origen y destino.
    No es la que quiere la maestra
    """
    
    A=[]
    for i in range(source_points.shape[0]):
        x,y=source_points[i,0],source_points[i,1]
        xw,yw=destination_points[i,0],destination_points[i,1]
        A.append([x,y,1,0,0,0,-xw*x,-xw*y,-xw])
        A.append([0,0,0,x,y,1,-yw*x,-yw*y,-yw])
    A=np.array(A)
    eigenvalues, eigenvectors = np.linalg.eig(A.T@A)
    min_eig_idx=np.argmin(eigenvalues)
    smallest_eigen_vector=eigenvectors[:,min_eig_idx]
    H=np.reshape(smallest_eigen_vector,(3,3))
    H=H/H[2,2]
    return H


def homografia(puntos_origen, puntos_destino):
    A = []
    for i in range(4): 
        x, y = puntos_origen[i]
        x_prima, y_prima = puntos_destino[i]
        A.append([x, y, 1, 0, 0, 0, -x_prima * x, -x_prima * y])
        A.append([0, 0, 0, x, y, 1, -y_prima * x, -y_prima * y])
    
    A = np.array(A) 
    
    b = np.array([puntos_destino[0, 0], puntos_destino[0, 1],
                 puntos_destino[1, 0], puntos_destino[1, 1],
                 puntos_destino[2, 0], puntos_destino[2, 1],
                 puntos_destino[3, 0], puntos_destino[3, 1]])
    
    h = np.linalg.solve(A, b)
    
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0]
    ])
    return H


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        param.append([x, y])



def save_coordinates(puntos, filename):
    with open(filename, "w") as f:
        json.dump(puntos, f)


def load_coordinates(filename):
    with open(filename, "r") as f:
        puntos = json.load(f)
    return [tuple(p) for p in puntos]

# =============================================================================
# MARCAR PUNTOS
# =============================================================================

def marcar(img):
    puntos = []
    cv2.namedWindow("Imagen")
    cv2.setMouseCallback("Imagen", mouse_callback, puntos)
    
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or len(puntos) >= 4:
            break

        temp_img = img.copy()
        for corner in puntos:
            x, y = corner
            cv2.circle(temp_img, (x, y), 7, (0, 0, 255), -1)
        cv2.imshow("Imagen", temp_img)

    for corner in puntos:
        x, y = corner
        cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
    cv2.imshow("Imagen", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return puntos


def new_corditates():
    puntos_img1 = marcar(img_1.copy())
    puntos_img2 = marcar(img_2.copy())
    save_coordinates(puntos_img1, coords_file1)
    save_coordinates(puntos_img2, coords_file2)



img_1 = lista_natural_imgs[0]
img_2 = lista_natural_imgs[1]

coords_file1 = os.path.join(ruta, "coordenadas_img1.json")
coords_file2 = os.path.join(ruta, "coordenadas_img2.json")

if os.path.exists(coords_file1) and os.path.exists(coords_file2):
    puntos_img1 = load_coordinates(coords_file1)
    puntos_img2 = load_coordinates(coords_file2)
    os.system("clear")
    print("Puntos cargados.")
    nuevas = input("¿Deseas marcar nuevas coordenadas? (s/n): ")
    if nuevas.lower() == 's':
        new_corditates()
        print("Nuevas coordenadas guardadas.")
else:
    new_corditates()
    print("Coordenadas guardadas.")

print("Coordenadas definitivas:")
print("Imagen 1:", puntos_img1)
print("Imagen 2:", puntos_img2)
 
# Verificar que se hayan marcado 4 puntos en cada imagen
if len(puntos_img1) != 4 or len(puntos_img2) != 4:
    print("Error: Se requieren 4 puntos en cada imagen para calcular la homografía.")
    exit(1)

# resultados matriz de homografía
H = homografia(np.array(puntos_img1), np.array(puntos_img2))

print("MATRIZ MANUAL")
print(H)

print("MATRIZ CV2")
print(cv2.findHomography(np.array(puntos_img1), np.array(puntos_img2)))



# =============================================================================
# CLASE PARA REALIZAR LA TRANSFORMACIÓN MANUAL PIXEL A PIXEL
# =============================================================================
class ManualPanoramaWarper:
    def __init__(self, base_image, image_to_warp, H):
        """
        base_image: Imagen base (img₂, donde recuperas los puntos) que usaremos como canvas.
        image_to_warp: Imagen a transformar (img₁, donde hiciste los puntos).
        H: Matriz de homografía que transforma image_to_warp a las coordenadas de base_image.
           Se calculó como H = homografia(puntos_img1, puntos_img2).

        """
        self.base_image = base_image
        self.image_to_warp = image_to_warp
        self.H = H
        self.h_warp, self.w_warp = image_to_warp.shape[:2]
        self.h_base, self.w_base = base_image.shape[:2]

    def transform_point(self, x, y):
        # Transforma el punto (x, y) de image_to_warp usando la matriz H.
        p = np.array([x, y, 1])
        p_t = self.H @ p
        p_t /= p_t[2]  # Normalizamos
        return p_t[0], p_t[1]

    def warp(self):
        # Usamos la imagen base (img₂) como canvas.
        panorama = self.base_image.copy()
        # Recorremos image_to_warp pixel a pixel.
        for y in range(self.h_warp):
            for x in range(self.w_warp):
                # Si el píxel tiene contenido (no es negro)
                if not np.all(self.image_to_warp[y, x] == 0):
                    new_x, new_y = self.transform_point(x, y)
                    new_x = int(round(new_x))
                    new_y = int(round(new_y))
                    # Solo asignamos si el nuevo píxel cae dentro de los límites del canvas.
                    if 0 <= new_x < self.w_base and 0 <= new_y < self.h_base:
                        panorama[new_y, new_x] = self.image_to_warp[y, x]
        return panorama

# =============================================================================
# USAR LA CLASE PARA GENERAR LA SEMI-PANORÁMICA MANUAL
# =============================================================================
warper = ManualPanoramaWarper(img_2, img_1, H)
panorama_manual = warper.warp()

# Mostrar el resultado
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(panorama_manual, cv2.COLOR_BGR2RGB))
plt.title("Semi-Panorámica Manual Pixel a Pixel 🖼️")
plt.axis("off")
plt.savefig("imagen_panoramica.png", dpi=300, bbox_inches="tight")
plt.show()