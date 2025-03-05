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

