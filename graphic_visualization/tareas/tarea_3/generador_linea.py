import numpy as np




tamano = 20

cord_uno = [2, 4]
cord_dos = [12, 18]



matriz = np.array([["░" for i in range(tamano)] for i in range(tamano)], dtype=str)


for fila in matriz:
    print(" ".join(fila))
    
print("\n")


columnas = len(matriz[0])
filas = len(matriz)


def pintar_pixel(x,y):
    return [x,y] == cord_uno





for i,_ in enumerate(range(filas)):
    for j,_ in enumerate(range(filas)):
        if pintar_pixel(i,j):
            matriz[i][j] = "█"


for fila in matriz:
    print(" ".join(fila))