import pennylane as qml
from pennylane import numpy as np

# el pip install es "pip install pennylane"

# Configuramos el dispositivo cuántico
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Definimos nuestro circuito cuántico
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Codificamos los datos en los qubits
    qml.RX(inputs[0], wires=0)
    qml.RX(inputs[1], wires=1)
    # Circuito variacional: parámetros entrenables
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    # Creamos entrelazamiento
    qml.CNOT(wires=[0,1])
    # Medimos el observable para obtener una salida
    return qml.expval(qml.PauliZ(0))

# Ejemplo de ejecución
weights = np.random.random(2)
result = quantum_circuit(np.array([0.5, 0.1]), weights)
print("Resultado del circuito:", result)