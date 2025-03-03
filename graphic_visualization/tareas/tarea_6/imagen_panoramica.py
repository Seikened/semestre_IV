import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

ruta =  os.path.join(os.path.dirname(__file__), "img")

img1 = cv2.imread(os.path.join(ruta, "img1.jpeg"))
img2 = cv2.imread(os.path.join(ruta, "img2.jpeg"))



# Pasamos las imagenes a escala de grise
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

'''
existe una funcion de cv2 que se llama ORB que nos permite detectar los puntos clave de una imagen
'''
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1_gray, None)
kp2, des2 = orb.detectAndCompute(img2_gray, None)
print(f"Numero de puntos clave en la imagen 1: {len(kp1)} | Descriptores: {des1.shape}")
print(f"Numero de puntos clave en la imagen 2: {len(kp2)} | Descriptores: {des2.shape}")



bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)
good_matches = []
for m,n  in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
print(f"Numero de puntos clave emparejados: {len(good_matches)}")

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
print(f"Matriz de homografia:\n{H}")

alto, ancho = img2.shape[:2]

resultado = cv2.warpPerspective(img1, H, (ancho, alto))
resultado[0:alto, 0:ancho] = img2


# Vamos a dibujar los puntos clave en las imagenes
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
plt.title("Imagen panoramica")
plt.axis("off")
plt.show()