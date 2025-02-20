import pygame
from rich.console import Console
from rich.table import Table
import numpy as np
import enum
import sys

console = Console()


class Confi(enum.Enum):
    tamano = 50
    tam_pix = 10
    cord_uno = [2, 4]    
    cord_dos = [48, 20]
    blanco = (255,255,255)
    gris = (200,200,200)
    color_pixel = (155, 89, 182)
    

def crear_matriz(ancho, alto):
    return np.array([["░" for _ in range(ancho)] for _ in range(alto)], dtype=str)

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



def circulo(cx,cy, r):
    circulo = []
    x = 0
    y = -r
    p = -r
    while x < -y:
        if p > 0:
            y += 1
            p += 2*(x+y) + 1
        else:
            p += 2*x + 1    
        
        circulo.append([cx + x, cy + y])
        circulo.append([cx - x, cy + y])
        circulo.append([cx + x, cy - y])
        circulo.append([cx - x, cy - y])
        circulo.append([cx + y, cy + x])
        circulo.append([cx + y, cy - x])
        circulo.append([cx - y, cy + x])
        circulo.append([cx - y, cy - x])
        
        x += 1
        
    return circulo
        
    



pygame.init()

v_an = 800
v_al = 600

ventana = pygame.display.set_mode((v_an,v_al))
pygame.display.set_caption("Hola fer")

punto_inicial = None

def dibujar_rejilla():
    for x in range(0,v_an,Confi.tam_pix.value):
        pygame.draw.line(ventana,Confi.gris.value,(x,0),(x,v_al))
    for y in range(0,v_al,Confi.tam_pix.value):
        pygame.draw.line(ventana,Confi.gris.value,(0,y),(v_an,y))

def dibujar_pixel(x,y,color= Confi.color_pixel.value):
    pygame.draw.rect(ventana,color,(x*Confi.tam_pix.value,y*Confi.tam_pix.value,Confi.tam_pix.value,Confi.tam_pix.value))


while True:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif evento.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            punto_inicial = (x // Confi.tam_pix.value, y // Confi.tam_pix.value)

    ventana.fill(Confi.blanco.value)
    dibujar_rejilla()

    if punto_inicial is not None:
        # Obtenemos la posición actual del mouse
        x, y = pygame.mouse.get_pos()
        mouse = (x // Confi.tam_pix.value, y // Confi.tam_pix.value)

        # Creamos la matriz y aplicamos el algoritmo de Bresenham
        matriz = crear_matriz(ventana.get_width() // Confi.tam_pix.value, ventana.get_height() // Confi.tam_pix.value)
        matriz = bresenham(matriz, punto_inicial, mouse)

        # Dibujamos solo los píxeles marcados con "█"
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                if matriz[i][j] == "█":
                    dibujar_pixel(j, i)

    pygame.display.flip()
    
# circulos = []

# while True:
#     for evento in pygame.event.get():
#         if evento.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()
#         elif evento.type == pygame.MOUSEBUTTONDOWN:
#             x, y = pygame.mouse.get_pos()
#             grid_x = x // Confi.tam_pix.value
#             grid_y = y // Confi.tam_pix.value
#             # Almacena los puntos del círculo al hacer clic
#             puntos_circulo = circulo(grid_x, grid_y, r=6)
#             circulos.append(puntos_circulo)

#     ventana.fill(Confi.blanco.value)
#     dibujar_rejilla()

#     # Dibuja todos los círculos almacenados
#     for puntos_circulo in circulos:
#         for (j, i) in puntos_circulo:
#             dibujar_pixel(j, i)

#     pygame.display.flip()


