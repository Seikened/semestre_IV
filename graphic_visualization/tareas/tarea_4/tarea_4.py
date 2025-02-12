from rich.console import Console
from rich.table import Table
import numpy as np

console = Console()

def crear_matriz(tamano):
    return np.array([["░" for _ in range(tamano)] for _ in range(tamano)], dtype=str)

def imprimir_matriz(matriz):
    table = Table(show_header=False, show_lines=False, expand=True)
    for fila in matriz:
        table.add_row(" ".join(f"[bold green]{c}[/]" if c == "█" else f"[dim]{c}[/]" for c in fila))
    console.print(table)

def bresenham(matriz, p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        matriz[y1][x1] = "█"  # Dibujo
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return matriz


tamano = 50
cord_uno = [2, 4]    
cord_dos = [48, 20]  

matriz = crear_matriz(tamano)
matriz = bresenham(matriz, cord_uno, cord_dos)
imprimir_matriz(matriz)