import numpy as np

def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def extract_features(signal):
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "max": np.max(signal),
        "min": np.min(signal)
    }

if __name__ == "__main__":
    signal = np.random.rand(100)
    print(extract_features(signal))
