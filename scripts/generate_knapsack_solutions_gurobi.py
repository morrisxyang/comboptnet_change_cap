import argparse
import os
import time
from typing import Tuple

import numpy as np


def load_instances(dataset_dir: str) -> np.ndarray:
    # instances_path = os.path.join(dataset_dir, "test_instances.npy")
    instances_path = os.path.join(dataset_dir, "train_instances.npy")
    if not os.path.isfile(instances_path):
        raise FileNotFoundError(f"Missing file: {instances_path}")
    instances = np.load(instances_path, allow_pickle=False)  # shape: (N, 10, 2) => [weight, price]
    if instances.ndim != 3 or instances.shape[2] != 2:
        raise ValueError(f"Expected instances shape (N, 10, 2). Got {instances.shape}")
    return instances


def solve_knapsack_gurobi(weights: np.ndarray, prices: np.ndarray, capacity: float) -> Tuple[np.ndarray, float]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError(
            "Failed to import gurobipy. Ensure Gurobi is installed and licensed."
        ) from e

    n = weights.shape[0]
    model = gp.Model()
    model.Params.OutputFlag = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(prices[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity")

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find optimal solution. Status: {model.status}")

    x_opt = np.array([int(round(x[i].X)) for i in range(n)], dtype=np.int64)
    obj = float(model.objVal)
    return x_opt, obj


def main():
    parser = argparse.ArgumentParser(description="Generate knapsack solutions with Gurobi for a given capacity")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "knapsack"),
        help="Directory containing test_instances.npy",
    )
    parser.add_argument("--capacity", type=float, default=150.0, help="Knapsack capacity")
    parser.add_argument("--preview", type=int, default=5, help="Print details for first N instances")
    args = parser.parse_args()

    instances = load_instances(args.dataset_dir)
    n_instances, n_items, _ = instances.shape

    solutions = np.zeros((n_instances, n_items), dtype=np.int64)
    objectives = np.zeros(n_instances, dtype=np.float64)
    weights_used = np.zeros(n_instances, dtype=np.float64)

    t0 = time.time()
    for i in range(n_instances):
        wp = instances[i]
        weights = wp[:, 0].astype(float)
        prices = wp[:, 1].astype(float)
        x_opt, obj = solve_knapsack_gurobi(weights, prices, args.capacity)
        solutions[i] = x_opt
        objectives[i] = obj
        weights_used[i] = float(np.dot(weights, x_opt))

        if i < args.preview:
            print(f"Instance {i} | items: {n_items}")
            print(f"  capacity: {args.capacity:.6f}")
            print(f"  used_weight: {weights_used[i]:.6f}")
            print(f"  objective: {objectives[i]:.6f}")
            print(f"  chosen_items: {int(np.sum(x_opt))}")
            print(f"  solution: {x_opt}")

    elapsed = time.time() - t0

    cap_int = int(round(args.capacity))
    # save_path = os.path.join(args.dataset_dir, f"test_sols_cap{cap_int}.npy")
    save_path = os.path.join(args.dataset_dir, f"train_sols_cap{cap_int}.npy")
    np.save(save_path, solutions)

    print("")
    print(f"Saved solutions: {save_path}")
    print(f"Instances solved: {n_instances}")
    print(f"Capacity: {args.capacity:.6f}")
    print(f"Avg objective: {objectives.mean():.6f} (min {objectives.min():.6f}, max {objectives.max():.6f})")
    print(f"Avg used weight: {weights_used.mean():.6f} (min {weights_used.min():.6f}, max {weights_used.max():.6f})")
    print(f"Avg items chosen: {np.mean(np.sum(solutions, axis=1)):.6f}")
    print(f"Total time: {elapsed:.3f}s | per-instance: {elapsed / n_instances:.6f}s")


if __name__ == "__main__":
    main()


