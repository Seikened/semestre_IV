

# PanorámicaV2

Este proyecto genera una panorámica a partir de dos imágenes. Se marcan puntos en las imágenes para calcular la homografía y transformar la imagen de origen al canvas de la imagen base.

---

## Requisitos

- **Python 3**
- **OpenCV**
- **NumPy**
- **Matplotlib**

Instala las dependencias con:

```bash
pip install opencv-python numpy matplotlib
```

---

## Uso

1. **Coloca las imágenes**:  
   Pon las imágenes (`img1.jpeg`, `img2.jpeg`, etc.) en la carpeta `img` que se encuentra en el mismo directorio que `panoramicaV2.py`.

2. **Ejecuta el script**:

   ```bash
   python panoramicaV2.py
   ```

3. **Marcado y carga de puntos**:  
   - Si no hay archivos de coordenadas, se te pedirá marcar **4 puntos** en cada imagen.
   - Si ya existen, se cargarán y se te preguntará si deseas marcarlas de nuevo.
   - **Importante:** Debes marcar exactamente 4 puntos en cada imagen para calcular la homografía.

4. **Visualiza el resultado**:  
   El script mostrará la imagen panorámica y guardará un gráfico en `grafico.png`.

