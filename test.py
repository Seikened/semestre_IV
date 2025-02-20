import time
import os

def crear_matriz(ancho, alto):
    """Crea una matriz vacía con puntos como fondo"""
    return [["." for _ in range(ancho)] for _ in range(alto)]

def imprimir_bubble(matriz, paso, x, y, err):
    """Imprime la matriz en un marco estilo 'bubble' junto con los valores actuales"""
    ancho = len(matriz[0])
    borde_superior = "┌" + "─" * (ancho * 2 - 1) + "┐"
    borde_inferior = "└" + "─" * (ancho * 2 - 1) + "┘"
    
    print(f"\nPaso {paso}: (x={x}, y={y}, err={err})")
    print(borde_superior)
    for fila in matriz:
        print("│" + " ".join(fila) + "│")
    print(borde_inferior)
    print("\n" + "=" * (ancho * 2 + 3) + "\n")

def bresenham_animado(ancho, alto, p1, p2):
    """Ejecuta el algoritmo de Bresenham y lo reinicia en un bucle infinito"""
    while True:  # Bucle infinito
        matriz = crear_matriz(ancho, alto)
        x1, y1 = p1
        x2, y2 = p2

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        paso = 0
        while True:
            # Dibuja el píxel actual en la matriz
            matriz[y1][x1] = "█"
            # Imprime el estado actual en un marco (bubble)
            imprimir_bubble(matriz, paso, x1, y1, err)
            time.sleep(0.5)  # Pausa para ver la animación

            # Si se llegó al punto final, sale del bucle interno
            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

            paso += 1

        # Reinicia el dibujo desde cero tras completar la línea
        print("\n🔄 Reiniciando en 2 segundos...\n")
        time.sleep(2)  # Espera antes de reiniciar

# Parámetros del ejemplo
ANCHO = 20
ALTO = 20
P1 = (2, 2)   # Punto inicial
P2 = (5, 17)  # Punto final

bresenham_animado(ANCHO, ALTO, P1, P2)