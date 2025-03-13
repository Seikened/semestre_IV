import matplotlib.pyplot as plt
import numpy as np
import cv2
import os



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




def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        param.append([x, y])


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


puntos_img1 = marcar(lista_natural_imgs[0].copy())
puntos_img2 = marcar(lista_natural_imgs[1].copy())

# resultados matriz de homografía
H = homography(np.array(puntos_img1), np.array(puntos_img2))

print(puntos_img1)
print(puntos_img2)
print(H)

# Warpeamos la imagen 1 para alinearla con la imagen 2
warp_image = cv2.warpPerspective(
    lista_natural_imgs[0],
    H,
    (lista_natural_imgs[0].shape[1] + lista_natural_imgs[1].shape[1],
     lista_natural_imgs[0].shape[0])
)
gray_img = cv2.cvtColor(warp_image, cv2.COLOR_BGR2LAB)
maskw = cv2.inRange(gray_img, np.array([0, 128, 128]), np.array([255, 255, 255]))

canvas_img2 = np.zeros_like(warp_image)
canvas_img2[0:lista_natural_imgs[1].shape[0], 0:lista_natural_imgs[1].shape[1]] = lista_natural_imgs[1]

canvas_img2[maskw == 0] = np.array([0, 0, 0])
# Y se pone en negro warp_image donde maskw es 255
warp_image[maskw == 255] = np.array([0, 0, 0])

# Se realiza la mezcla de las imágenes
blend_image = cv2.add(warp_image, (canvas_img2 * 0.9).astype(np.uint8))
panorama = warp_image.copy()
mask_warp = cv2.cvtColor(warp_image, cv2.COLOR_BGR2GRAY)
mask_canvas = cv2.cvtColor(canvas_img2, cv2.COLOR_BGR2GRAY)
_, mask_warp_bin = cv2.threshold(mask_warp, 1, 255, cv2.THRESH_BINARY)
_, mask_canvas_bin = cv2.threshold(mask_canvas, 1, 255, cv2.THRESH_BINARY)
mask_warp_bool = mask_warp_bin.astype(bool)
mask_canvas_bool = mask_canvas_bin.astype(bool)
overlap = mask_warp_bool & mask_canvas_bool
only_warp = mask_warp_bool & ~mask_canvas_bool  # Sólo contiene warp_image
only_canvas = mask_canvas_bool & ~mask_warp_bool  # Sólo contiene canvas_img2
panorama[only_warp] = warp_image[only_warp]
panorama[only_canvas] = canvas_img2[only_canvas]
blended = cv2.addWeighted(warp_image, 0.5, canvas_img2, 0.5, 0)
panorama[overlap] = blended[overlap]

# Mostramos la panorámica final
cv2.imshow("Panorámica", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()