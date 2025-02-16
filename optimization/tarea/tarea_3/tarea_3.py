import time
import functools
import random
import bisect
from tabulate import tabulate  

'''
Tarea 02

1. Busca un algoritmo correspondiente a cada complejidad (ver Tabla Diap 17) y analiza su tiempo de computo T(n), su complejidad (Mejor, peor, y caso promedio)

2. Implementalos usando Python (sin Numpy, usa funciones) y toma su tiempo de computo en tu máquina para diferentes tamaños de entrada. (Realiza una tabla parecida a la de la Diap 17: Algoritmo vs n/tiempo (s))

3. Recuerda reportar tus consultas de IA adjuntando tus prompts al reporte.

4. Recuerda también la nomenclatura de las tareas

'''

def princ(mensaje, color="pastel_blue"):
    """
    Imprime el mensaje en un color pastel especificado.
    Colores disponibles:
        - pastel_red
        - pastel_green
        - pastel_yellow
        - pastel_blue
        - pastel_magenta
        - pastel_cyan
    """
    # Diccionario con códigos ANSI para colores pastel (utilizando variantes brillantes)
    pastel_colors = {
        "pastel_red": "\033[91m",
        "pastel_green": "\033[92m",
        "pastel_yellow": "\033[93m",
        "pastel_blue": "\033[94m",
        "pastel_magenta": "\033[95m",
        "pastel_cyan": "\033[96m"
    }
    # Se obtiene el código ANSI para el color elegido, por defecto pastel_blue
    color_code = pastel_colors.get(color, "\033[94m")
    # Se imprime el mensaje en el color deseado y luego se resetea el color
    print(f"{color_code}{mensaje}\033[0m")

def generdor_lista(n, numero_incluido):
    lista = [random.randint(0, n) for _ in range(n)]
    indice = random.randint(0, n-1)
    lista[indice] = numero_incluido 
    random.shuffle(lista)
    return lista


time_results = {}  

def imprimir_tabla_tiempos():
    tabla = []
    for algoritmo, datos in time_results.items():
        for tam, caso, tiempo in datos:
            tabla.append([algoritmo, tam, caso, f"{tiempo}"])
    
    print("\nTabla de tiempos:")
    print(tabulate(tabla, headers=["Algoritmo", "Tamaño", "Caso", "Tiempo (s)"], tablefmt="pretty"))


class Algoritmo:



   
    
    @staticmethod
    def medidor_tiempo(funcion):
        @functools.wraps(funcion)
        def wrapper(*args, **kwargs):
            if getattr(wrapper, "en_ejecucion", False):
                return funcion(*args, **kwargs)
            else:
                wrapper.en_ejecucion = True
                # Usamos time.perf_counter() para mayor precisión
                inicio = time.perf_counter()
                resultado = funcion(*args, **kwargs)
                fin = time.perf_counter()
                tiempo_total = fin - inicio
                # Redondeamos a 8 decimales siempre (sin condicional)
                tiempo_segundos = round(tiempo_total, 8)
                
                # Extraer el tamaño y el caso, si se pasan
                tamano = kwargs.get("tamano", "N/A")
                caso = kwargs.get("caso", "N/A")
                nombre_algo = funcion.__doc__ or funcion.__name__

                # Guardamos el tiempo en el diccionario global
                if nombre_algo not in time_results:
                    time_results[nombre_algo] = []
                time_results[nombre_algo].append((tamano, caso, tiempo_segundos))
                
                princ("-"*100, "pastel_yellow")
                princ(f"Algoritmo: {nombre_algo}", "pastel_cyan")
                princ(f"Caso: {caso} | Tamaño: {tamano}", "pastel_green")
                princ(f"Tiempo de ejecución: {tiempo_segundos} segundos", "pastel_magenta")
                wrapper.en_ejecucion = False
                return resultado
        wrapper.en_ejecucion = False
        return wrapper
    
    
    
    @medidor_tiempo
    def o_constante(self):
        """Algoritmo 1/8: T(n) = O(1) - Impresión de una lista"""
        lista = ["1"]
        n = len(lista) # c1
        for i in range(n): # c2
            print(i)
        return n # c3
        # T(n) = c1 + c2 + c3

    @medidor_tiempo
    def busqueda_binaria(self,lista, valor):
        """Algoritmo 2/8: T(n) = O(log n) - Búsqueda binaria"""
        inicio = 0 # c1
        fin = len(lista) - 1 # c2
        while inicio <= fin: # n/2 c3
            medio = (inicio + fin) // 2 # c4
            if lista[medio] == valor: # c5
                return medio # c6
            elif lista[medio] < valor: # c7
                inicio = medio + 1 # c8
            else: # c9
                fin = medio - 1 # c10
        return None # c11
        # T(n) = c1 + c2 + n/2 c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11
        

    @medidor_tiempo
    def busqueda_secuencial(self,lista, valor):
        """Algoritmo 3/8: T(n) = O(n) - Búsqueda secuencial"""
        for i in range(len(lista)): # n c1
            if lista[i] == valor: # c2
                return i # c3
        return None # c4
        # T(n) = n c1 + c2 + c3 + c4

    @medidor_tiempo
    def o_nlogn(self,lista):
        """Algoritmo 4/8: T(n) = O(n log n) - Ordenamiento quicksort"""
        if len(lista) <= 1: # c1
            return lista # c2
        
        pivote = lista[0] # c3
        
        menos = [x for x in lista[1:] if x < pivote]  # n
        mas = [x for x in lista[1:] if x >= pivote] # n
        return self.o_nlogn(menos) + [pivote] + self.o_nlogn(mas)  # n/2^k
        # T(n) = c1 + c2 + c3 + n + n + n/2^k

    @medidor_tiempo
    def ordenamiento_burbuja(self,lista):
        """Algoritmo 5/8: T(n) = O(n²) - Ordenamiento burbuja"""
        n = len(lista) # c1
        for i in range(0,n-1): # (n-1)c2 
            for j in range(0, n - i - 2): # (n-2)c3
                if lista[j] > lista[j + 1]: # c4
                    lista[j], lista[j + 1] = lista[j + 1], lista[j] # c5
        return lista # c6
        # T(n) = c1 + (n-1)c2 + (n-2)c3 + c4 + c5 + c6

    @medidor_tiempo
    def o_n_cubo(self,matriz1, matriz2):
        """Algoritmo 6/8: T(n) = O(n³) - Multiplicación de matrices"""
        n = len(matriz1) # c1
        resultado =  [[0] * n  for _ in range(n)] # n
        for i in range(n): # n
            for j in range(n): # n 
                for k in range(n): # n
                    resultado[i][j] += matriz1[i][k] * matriz2[k][j] # c2
        return resultado # c3

    @medidor_tiempo
    def o_exponencial(self,n):
        """Algoritmo 7/8: T(n) = O(2^n) - Subconjuntos de un conjunto"""
        if n ==0: # c1
            return [[]] # c2
        subsets = self.o_exponencial(n-1) # c3
        return subsets + [s + [n-1] for s in subsets] # c4


    @medidor_tiempo
    def o_factorial(self,lista):
        """Algoritmo 8/8: T(n) = O(n!) - Permutaciones de una lista"""
        if not lista:
            return [[]]
        result = []
        for i in range(len(lista)):
            resto = lista[:i] + lista[i+1:]
            for perm in self.o_factorial(resto):
                result.append([lista[i]] + perm)
        return result

    def prueba(self,lista,valor,matris):
        self.o_constante()
        self.busqueda_binaria(lista, valor)
        self.busqueda_secuencial(lista, valor)
        self.o_nlogn(lista)
        self.ordenamiento_burbuja(lista)
        self.o_n_cubo(matris, matris)
        self.o_exponencial(len(lista))
        self.o_factorial(lista)



    


def uni_test(tamano):
    algoritmo = Algoritmo()

    # -----------------------------
    # Caso 1: PEOR CASO - Todo es aleatorio
    # -----------------------------
    valor_peor = random.randint(0, tamano)
    lista_peor = generdor_lista(tamano, valor_peor)
    matriz_peor = [generdor_lista(tamano, valor_peor) for _ in range(tamano)]
    print("\n")
    princ("CASO PEOR: Todo aleatorio", "pastel_red")
    algoritmo.o_constante(tamano=tamano, caso="PEOR")
    algoritmo.busqueda_binaria(lista_peor, valor_peor, tamano=tamano, caso="PEOR")
    algoritmo.busqueda_secuencial(lista_peor, valor_peor, tamano=tamano, caso="PEOR")
    algoritmo.o_nlogn(lista_peor, tamano=tamano, caso="PEOR")
    algoritmo.ordenamiento_burbuja(lista_peor, tamano=tamano, caso="PEOR")
    algoritmo.o_n_cubo(matriz_peor, matriz_peor, tamano=tamano, caso="PEOR")
    algoritmo.o_exponencial(len(lista_peor), tamano=tamano, caso="PEOR")
    algoritmo.o_factorial(lista_peor, tamano=tamano, caso="PEOR")

    # -----------------------------
    # Caso 2: MEJOR CASO - Lista ordenada
    # -----------------------------
    lista_mejor = sorted([random.randint(0, tamano) for _ in range(tamano)])
    valor_mejor_seq = lista_mejor[0]
    valor_mejor_bin = lista_mejor[len(lista_mejor)//2]
    matriz_mejor = [sorted([random.randint(0, tamano) for _ in range(tamano)]) for _ in range(tamano)]
    print("\n")
    princ("CASO MEJOR: Lista ordenada", "pastel_green")
    algoritmo.o_constante(tamano=tamano, caso="MEJOR")
    algoritmo.busqueda_secuencial(lista_mejor, valor_mejor_seq, tamano=tamano, caso="MEJOR")
    algoritmo.busqueda_binaria(lista_mejor, valor_mejor_bin, tamano=tamano, caso="MEJOR")
    algoritmo.o_nlogn(lista_mejor, tamano=tamano, caso="MEJOR")
    algoritmo.ordenamiento_burbuja(lista_mejor, tamano=tamano, caso="MEJOR")
    algoritmo.o_n_cubo(matriz_mejor, matriz_mejor, tamano=tamano, caso="MEJOR")
    algoritmo.o_exponencial(len(lista_mejor), tamano=tamano, caso="MEJOR")
    algoritmo.o_factorial(lista_mejor, tamano=tamano, caso="MEJOR")

    # -----------------------------
    # Caso 3: PROMEDIO - Lista semi-ordenada
    # -----------------------------
    mitad = tamano // 2
    primera_mitad = sorted([random.randint(0, tamano) for _ in range(mitad)])
    segunda_mitad = [random.randint(0, tamano) for _ in range(tamano - mitad)]
    lista_promedio = primera_mitad + segunda_mitad
    valor_promedio = random.randint(0, tamano)
    indice_promedio = len(lista_promedio) // 2
    lista_promedio[indice_promedio] = valor_promedio
    matriz_promedio = []
    for _ in range(tamano):
         mitad_matriz = tamano // 2
         primera_matriz = sorted([random.randint(0, tamano) for _ in range(mitad_matriz)])
         segunda_matriz = [random.randint(0, tamano) for _ in range(tamano - mitad_matriz)]
         fila = primera_matriz + segunda_matriz
         matriz_promedio.append(fila)
         
    print("\n")
    princ("CASO PROMEDIO: Lista semi-ordenada", "pastel_yellow")
    algoritmo.o_constante(tamano=tamano, caso="PROMEDIO")
    algoritmo.busqueda_secuencial(lista_promedio, valor_promedio, tamano=tamano, caso="PROMEDIO")
    algoritmo.busqueda_binaria(lista_promedio, valor_promedio, tamano=tamano, caso="PROMEDIO")
    algoritmo.o_nlogn(lista_promedio, tamano=tamano, caso="PROMEDIO")
    algoritmo.ordenamiento_burbuja(lista_promedio, tamano=tamano, caso="PROMEDIO")
    algoritmo.o_n_cubo(matriz_promedio, matriz_promedio, tamano=tamano, caso="PROMEDIO")
    algoritmo.o_exponencial(len(lista_promedio), tamano=tamano, caso="PROMEDIO")
    algoritmo.o_factorial(lista_promedio, tamano=tamano, caso="PROMEDIO")
    



def multi_test():
    tamanos = [1,2,3,4,5,6,7,8,9,10]
    
    for tamano in tamanos:
        princ(f'PRUEBA DE TAMAÑO {tamano}', "pastel_blue")
        uni_test(tamano)

# ======= Uso de los algoritmos =========

'''
La funcion de multitest queda descontinuada por que solo soporta hasta el 10 de iteración....
'''

#multi_test()
uni_test(10)
imprimir_tabla_tiempos()