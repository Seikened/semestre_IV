from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
#from qiskit.providers.ibmq import IBMQ

import time
import os
import qiskit

print(f"Versión de quiskit: {qiskit.__version__}")




def medir_tiemo(func):
    def wrapper(*args, **kwargs):
        inicio = time.perf_counter()
        resultado = func(*args, **kwargs)
        fin = time.perf_counter()
        return resultado, round(fin - inicio,3)
    return wrapper

# Crear el circuito con 3 qubits y 3 bits clásicos
@medir_tiemo
def grover_algorithm():
    qc = QuantumCircuit(3, 3)

    # Inicializamos en superposición
    qc.h([0, 1, 2])
    qc.barrier()

    # Oráculo para marcar el estado |101⟩ (x0=1, x1=0, x2=1)
    qc.x(1)        # se invierte qubit 1 para ~x1
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x(1)
    qc.barrier()

    # Difusión (amplificación)
    qc.h([0, 1, 2])
    qc.x([0, 1, 2])
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x([0, 1, 2])
    qc.h([0, 1, 2])

    # Medición
    qc.measure([0, 1, 2], [0, 1, 2])

    # Ejecutamos en el simulador Aer
    backend = Aer.get_backend('qasm_simulator')
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=1024).result()
    counts = result.get_counts()

    return counts


# Busqueda
@medir_tiemo
def classical_search(target):
    # Creamos el espacio de búsqueda (para 3 bits: "000", "001", ..., "111")
    search_space = [format(i, '03b') for i in range(8)]
    for state in search_space:
        if state == target:
            return state
    return None

# Ejecutamos ambas búsquedas
grover_result, grover_time = grover_algorithm()
classical_result, classical_time = classical_search("101")

print("Resultados de Grover:", grover_result, "Tiempo:", grover_time, "segundos")
print("Resultado de búsqueda clásica:", classical_result, "Tiempo:", classical_time, "segundos")

# Mostrar histograma de Grover
plot_histogram(grover_result)
plt.title("Histograma de Resultados de Grover")
plt.xlabel("Estados")
plt.ylabel("Frecuencia")
plt.show()