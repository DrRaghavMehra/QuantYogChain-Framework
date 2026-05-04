import numpy as np

def load_dummy_wearable_data(n_samples=100):
    """
    Simulated dataset representing:
    HRV, EDA, SpO2, Sleep features
    """
    X = np.random.rand(n_samples, 8)  # 8 features
    y = np.random.randint(0, 2, size=(n_samples,))  # binary classification
    return X, y

if __name__ == "__main__":
    X, y = load_dummy_wearable_data()
    print("Data shape:", X.shape)
