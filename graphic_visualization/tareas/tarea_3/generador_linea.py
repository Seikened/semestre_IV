from rich.console import Console
from rich.table import Table
import numpy as np


console = Console()

tamano = 20
cord_uno = [2, 4]
cord_dos = [12, 18]



def crear_matriz(tamano):
    return np.array([["░" for _ in range(tamano)] for _ in range(tamano)], dtype=str)

def pintar_pixel(x, y,cord_uno,cord_dos):
    x_1,y_1 = cord_uno
    x_2,y_2 = cord_dos
    A = y_2 - y_1
    B = x_1 - x_2
    C = (x_2 * y_1) - (x_1*y_2)
    resultado = (A*x) + (B*y) + C 
    return 1 > resultado >= 0



def modificar_matriz(matriz):
    for i in range(tamano):
        for j in range(tamano):
            if pintar_pixel(i, j,cord_uno,cord_dos):
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
