from rich.console import Console
from rich.table import Table
import numpy as np


console = Console()

tamano = 50
cord_uno = [2, 4]
cord_dos = [48, 20]


def crear_matriz(tamano):
    return np.array([["░" for _ in range(tamano)] for _ in range(tamano)], dtype=str)


def pintar_pixel(x, y,cord_uno,cord_dos):
    x_1,y_1 = cord_uno
    x_2,y_2 = cord_dos
    A = y_2 - y_1
    B = x_1 - x_2
    C = (x_2 * y_1) - (x_1*y_2)
    resultado = abs((A*x) + (B*y) + C )
    return 1 > resultado >= 0



def modificar_matriz(matriz):
    for i in range(tamano):
        for j in range(tamano):
            if pintar_pixel(j, i,cord_uno,cord_dos):
                matriz[i][j] = "█"
    return matriz

def imprimir_matriz(matriz):
    table = Table(show_header=False, show_lines=False, expand=True)
    for fila in matriz:
        table.add_row(" ".join(f"[bold green]{c}[/]" if c == "█" else f"[dim]{c}[/]" for c in fila))
    console.print(table)

matriz = crear_matriz(tamano)
matriz = modificar_matriz(matriz)
imprimir_matriz(matriz)



# DDA

matriz = crear_matriz(tamano)


x_1,y_1 = cord_uno
x_2,y_2 = cord_dos

delta_X = abs(x_2 - x_1)
delta_Y = abs(y_2 - y_1)

pasos = max(delta_X,delta_Y)


X_inc = delta_X/pasos
Y_inc = delta_Y/pasos

x , y = x_1 , y_1


for _ in range(pasos):
    matriz[round(y)][round(x)] = "█"
    x += X_inc
    y += Y_inc

imprimir_matriz(matriz)