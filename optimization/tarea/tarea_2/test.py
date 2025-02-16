from bigO import BigO

# Función a analizar
def mi_funcion(arr):
    return sorted(arr)  # Esto debería tener complejidad O(n log n)

# Crear instancia de BigO
complejidad = BigO()

# Usamos un generador predefinido: integers para listas de enteros
resultado = complejidad.test(mi_funcion, "integer")

# Imprimir el resultado completo para ver su estructura
print(resultado)