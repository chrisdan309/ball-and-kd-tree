import numpy as np
import os

def generate_dataset(dimension: int, num_points: int, seed: int = 42):
    np.random.seed(seed)
    return np.random.rand(num_points, dimension)

def save_as_csv(data: np.ndarray, filename: str):
    np.savetxt(filename, data, delimiter=",", fmt="%.6f")

def save_as_npy(data: np.ndarray, filename: str):
    np.save(filename, data)

def generate_and_save_all(base_dir: str = "./tests/datasets", num_points: int = 1000):
    os.makedirs(base_dir, exist_ok=True)

    configs = [
        (2, "2D"),
        (10, "10D"),
        (50, "50D")
    ]

    for dim, label in configs:
        data = generate_dataset(dimension=dim, num_points=num_points)
        csv_path = os.path.join(base_dir, f"dataset_{label}.csv")
        npy_path = os.path.join(base_dir, f"dataset_{label}.npy")
        
        save_as_csv(data, csv_path)
        save_as_npy(data, npy_path)

        print(f"{label} dataset saved: {csv_path} and {npy_path}")

if __name__ == "__main__":
    generate_and_save_all()
