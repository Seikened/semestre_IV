import pygame
import sys
import math

ANCHO, ALTO = 800, 600

class CuerpoCeleste:
    def __init__(self, nombre, color, tamano, radio_orbita, velocidad_orbita, velocidad_rotacion, padre=None, angulo=0):
        self.nombre = nombre
        self.color = color
        self.tamano = tamano
        self.radio_orbita = radio_orbita
        self.velocidad_orbita = velocidad_orbita
        self.velocidad_rotacion = velocidad_rotacion
        self.padre = padre
        self.angulo = angulo
        self.angulo_rotacion = 0

        if self.padre is None:
            self.x = ANCHO / 2
            self.y = ALTO / 2
        else:
            self.x = 0
            self.y = 0

    def update(self):
        self.angulo_rotacion += self.velocidad_rotacion
        if self.padre:
            self.angulo += self.velocidad_orbita
            padre_x, padre_y = self.padre.x, self.padre.y
            self.x = padre_x + self.radio_orbita * math.cos(self.angulo)
            self.y = padre_y + self.radio_orbita * math.sin(self.angulo)

    def draw(self, pantalla):
        superficie = pygame.Surface((self.tamano, self.tamano), pygame.SRCALPHA)
        superficie.fill(self.color)
        superficie_rotada = pygame.transform.rotate(superficie, math.degrees(self.angulo_rotacion))
        rect = superficie_rotada.get_rect(center=(self.x, self.y))
        pantalla.blit(superficie_rotada, rect)

def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    reloj = pygame.time.Clock()

    sol = CuerpoCeleste("Sol", (255, 255, 0), 50, 0, 0, 0.01, None)
    tierra = CuerpoCeleste("Tierra", (0, 0, 255), 30, 150, 0.02, 0.05, sol)
    luna = CuerpoCeleste("Luna", (200, 200, 200), 15, 50, 0.05, 0.1, tierra)

    ejecutando = True
    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False

        pantalla.fill((0, 0, 0))
        sol.update()
        tierra.update()
        luna.update()
        sol.draw(pantalla)
        tierra.draw(pantalla)
        luna.draw(pantalla)
        pygame.display.flip()
        reloj.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
