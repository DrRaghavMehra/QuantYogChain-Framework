import pennylane as qml
from pennylane import numpy as np

# Number of qubits
n_qubits = 8

# Quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Variational Quantum Circuit
@qml.qnode(dev)
def vqc_circuit(x, weights):
    # Feature encoding
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

    # Variational layers
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Prediction function
def predict(x, weights):
    return vqc_circuit(x, weights)

if __name__ == "__main__":
    # Dummy test
    x = np.random.rand(n_qubits)
    weights = np.random.rand(3, n_qubits, 3)
    print(predict(x, weights))
