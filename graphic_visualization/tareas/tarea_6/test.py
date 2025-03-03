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

# Vamos a dibujar los puntos clave en las imagenes
img1_kp = cv2.drawKeypoints(img1_gray, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2_gray, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
plt.title("Imagen 1")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
plt.title("Imagen 2")
plt.show()