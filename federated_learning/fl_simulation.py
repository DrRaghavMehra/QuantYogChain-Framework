import numpy as np

class DummyModel:
    def __init__(self):
        self.weights = np.random.rand(10)

    def get_weights(self):
        return self.weights

    def set_weights(self, w):
        self.weights = w

    def train(self):
        self.weights += np.random.normal(0, 0.01, size=self.weights.shape)

# Simulate 3 clients
clients = [DummyModel() for _ in range(3)]

def federated_average(clients):
    weights = [c.get_weights() for c in clients]
    return np.mean(weights, axis=0)

if __name__ == "__main__":
    for round in range(5):
        for c in clients:
            c.train()

        global_weights = federated_average(clients)

        for c in clients:
            c.set_weights(global_weights)

        print(f"Round {round} completed")
