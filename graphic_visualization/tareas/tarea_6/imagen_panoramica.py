import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

ruta =  os.path.join(os.path.dirname(__file__), "img")

img1 = cv2.imread(os.path.join(ruta, "img1.jpeg"))
img2 = cv2.imread(os.path.join(ruta, "img2.jpeg"))
img3 = cv2.imread(os.path.join(ruta, "img3.jpeg"))
img4 = cv2.imread(os.path.join(ruta, "img4.jpeg"))
img5 = cv2.imread(os.path.join(ruta, "img5.jpeg"))
lista_natural_imgs = [img1, img2, img3, img4, img5]



# Pasamos las imagenes a escala de grise
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img5_gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
lista_gres_imgs = [img1_gray, img2_gray, img3_gray, img4_gray, img5_gray]


def unir_imagenes_modo_facil(imgs):
    """ Funciona para unir imagenes de manera facil """
    imagenes = imgs.copy()
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, resultado = stitcher.stitch(imagenes)
    if status != cv2.Stitcher_OK:
        print("No se pudo unir las imagenes")
        return None
    return resultado




'''
existe una funcion de cv2 que se llama ORB que nos permite detectar los puntos clave de una imagen
'''
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1_gray, None)
kp2, des2 = orb.detectAndCompute(img2_gray, None)
kp3, des3 = orb.detectAndCompute(img3_gray, None)
kp4, des4 = orb.detectAndCompute(img4_gray, None)
kp5, des5 = orb.detectAndCompute(img5_gray, None)

lista_kp_imgs = [kp1, kp2, kp3, kp4, kp5]
lista_des_imgs = [des1, des2, des3, des4, des5]





# Emparejar los descriptores de la imagen 2 y la imagen 3
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des2, des3, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
print(f"Numero de puntos clave emparejados: {len(good_matches)}")

# Extraer coordenadas: usar kp2 para la imagen 2 y kp3 para la imagen 3
pts_img2 = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts_img3 = np.float32([kp3[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calcular la homografía entre imagen 2 y 3
H, mask = cv2.findHomography(pts_img2, pts_img3, cv2.RANSAC, 5.0)
print(f"Matriz de homografia:\n{H}")

# Para visualizar la unión, usamos el tamaño de la imagen 3
alto, ancho = img3.shape[:2]

# Aplicar la transformación a la imagen 2 usando la homografía
resultado_manual = cv2.warpPerspective(img2, H, (ancho, alto))

resultado_manual[0:alto, 0:ancho] = img3

# Mostrar el resultado manual
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(resultado_manual, cv2.COLOR_BGR2RGB))
plt.title("Imagen panoramica manual (img2 e img3)")
plt.axis("off")
plt.show()



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


probar_stitcher_con_subconjunto(lista_natural_imgs, [0, 1])
probar_stitcher_con_subconjunto(lista_natural_imgs, [1, 2])
probar_stitcher_con_subconjunto(lista_natural_imgs, [2, 3])
probar_stitcher_con_subconjunto(lista_natural_imgs, [3, 4])