from quantum_model.vqc_model import predict
from data_processing.data_loader import load_dummy_wearable_data
import numpy as np

# Load data
X, y = load_dummy_wearable_data(50)

# Initialize weights
weights = np.random.rand(3, 8, 3)

# Run prediction
predictions = []
for x in X:
    output = predict(x, weights)
    predictions.append(np.argmax(output))

# Evaluate accuracy
accuracy = np.mean(np.array(predictions) == y)

print("Experiment Completed")
print("Accuracy:", accuracy)
