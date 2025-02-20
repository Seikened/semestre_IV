

class cuerpo_celeste:
    def __init__(self, nombre, masa, radio, distancia):
        self.nombre = nombre
        self.masa = masa
        self.radio = radio
        self.distancia = distancia

    def __str__(self):
        return f"{self.nombre} con masa {self.masa} kg, radio {self.radio} km y distancia {self.distancia} km"
    
