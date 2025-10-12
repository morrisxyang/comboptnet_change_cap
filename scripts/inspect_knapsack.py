import os
import argparse
import numpy as np


def summarize_array(name: str, arr: np.ndarray, sample_limit: int) -> None:
    print(f"\n=== {name} ===")
    print(f"shape: {arr.shape}, dtype: {arr.dtype}")
    if arr.size == 0:
        return
    # Basic stats for numeric arrays
    if np.issubdtype(arr.dtype, np.number):
        try:
            arr_flat = arr.reshape(-1)
            print(f"min: {arr_flat.min():.6f}, max: {arr_flat.max():.6f}, mean: {arr_flat.mean():.6f}, std: {arr_flat.std():.6f}")
        except Exception:
            pass

    # Show a few samples
    n = min(sample_limit, arr.shape[0]) if arr.ndim >= 1 else 1
    for i in range(n):
        sample = arr[i]
        if isinstance(sample, np.ndarray) and sample.ndim > 1:
            # For high-d arrays, show first row/cols succinctly
            preview = sample
            while preview.ndim > 2:
                preview = preview[0]
            # Truncate columns if very wide
            if preview.ndim == 2:
                rows = min(2, preview.shape[0])
                cols = min(10, preview.shape[1])
                print(f"sample[{i}] first {rows}x{cols}:\n{preview[:rows, :cols]}")
            else:
                cols = min(10, preview.shape[0])
                print(f"sample[{i}] first {cols}: {preview[:cols]}")
        else:
            # 1D or scalar
            if isinstance(sample, np.ndarray):
                cols = min(20, sample.shape[0]) if sample.ndim == 1 else 1
                print(f"sample[{i}] first {cols if sample.ndim == 1 else 1}: {sample[:cols] if sample.ndim == 1 else sample}")
            else:
                print(f"sample[{i}]: {sample}")


def check_binary_solutions(name: str, sols: np.ndarray) -> None:
    if sols.ndim == 0:
        return
    uniques = np.unique(sols)
    is_binary = set(uniques.tolist()).issubset({0, 1})
    print(f"\n[{name}] unique values: {uniques}")
    print(f"[{name}] binary: {is_binary}")
    # Show number of selected items for first few solutions
    limit = min(5, sols.shape[0])
    counts = [int(sols[i].sum()) for i in range(limit)]
    print(f"[{name}] sum(selected items) of first {limit}: {counts}")


def resolve_default_dataset_dir() -> str:
    # Default to repo-local datasets/knapsack relative to this file
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(repo_root, "data/original_datasets", "knapsack")


def main():
    parser = argparse.ArgumentParser(description="Inspect knapsack dataset npy files")
    parser.add_argument("--dataset-dir", type=str, default=resolve_default_dataset_dir(), help="Path to datasets/knapsack directory")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to preview per array")
    args = parser.parse_args()

    base = args.dataset_dir
    files = {
        "train_encodings": os.path.join(base, "train_encodings.npy"),
        "train_sols": os.path.join(base, "train_sols.npy"),
        "test_encodings": os.path.join(base, "test_encodings.npy"),
        "test_sols": os.path.join(base, "test_sols.npy"),
        "weights_prices": os.path.join(base, "weights_prices.npy"),
        "test_instances": os.path.join(base, "test_instances.npy"),
    }

    print(f"Dataset directory: {base}")
    for key, path in files.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    # Load
    train_enc = np.load(files["train_encodings"], allow_pickle=False)
    train_sols = np.load(files["train_sols"], allow_pickle=False)
    test_enc = np.load(files["test_encodings"], allow_pickle=False)
    test_sols = np.load(files["test_sols"], allow_pickle=False)
    weights_prices = np.load(files["weights_prices"], allow_pickle=False)
    test_instances = np.load(files["test_instances"], allow_pickle=False)


    # Summaries
    # summarize_array("train_encodings", train_enc, args.limit)
    # summarize_array("train_sols", train_sols, args.limit)
    # summarize_array("test_encodings", test_enc, args.limit)
    # summarize_array("test_sols", test_sols, args.limit)
    # summarize_array("weights_prices", weights_prices, args.limit)
    # summarize_array("test_instances", test_instances, args.limit)
    print(test_instances[0])
    print(test_sols[0])

    # Solution sanity checks
    # check_binary_solutions("train_sols", train_sols)
    # check_binary_solutions("test_sols", test_sols)
    #
    # # Cross-check shapes
    # if train_enc.shape[0] != train_sols.shape[0]:
    #     print(f"WARNING: train_encodings and train_sols count mismatch: {train_enc.shape[0]} vs {train_sols.shape[0]}")
    # if test_enc.shape[0] != test_sols.shape[0]:
    #     print(f"WARNING: test_encodings and test_sols count mismatch: {test_enc.shape[0]} vs {test_sols.shape[0]}")


if __name__ == "__main__":
    main()


