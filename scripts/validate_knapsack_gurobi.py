import argparse
import os
from typing import Tuple

import numpy as np


def load_data(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    # instances_path = os.path.join(dataset_dir, "test_instances.npy")
    # sols_path = os.path.join(dataset_dir, "test_sols_cap250.npy")
    instances_path = os.path.join(dataset_dir, "train_instances.npy")
    sols_path = os.path.join(dataset_dir, "train_sols_cap150.npy")
    if not os.path.isfile(instances_path):
        raise FileNotFoundError(f"Missing file: {instances_path}")
    if not os.path.isfile(sols_path):
        raise FileNotFoundError(f"Missing file: {sols_path}")
    instances = np.load(instances_path, allow_pickle=False)  # shape: (N, 10, 2) => [weight, price]
    sols = np.load(sols_path, allow_pickle=False)            # shape: (N, 10) => 0/1
    return instances, sols


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
    # Objective: maximize total price
    model.setObjective(gp.quicksum(prices[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    # Capacity constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity")

    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find optimal solution. Status: {model.status}")

    x_opt = np.array([int(round(x[i].X)) for i in range(n)], dtype=np.int64)
    obj = float(model.objVal)
    return x_opt, obj


def is_binary_vector(vec: np.ndarray, atol: float = 1e-8) -> bool:
    return np.all((np.isclose(vec, 0.0, atol=atol)) | (np.isclose(vec, 1.0, atol=atol)))


def validate(dataset_dir: str, capacity: float, atol_obj: float) -> int:
    instances, sols = load_data(dataset_dir)
    if instances.ndim != 3 or instances.shape[2] != 2:
        raise ValueError(f"Expected instances shape (N, 10, 2). Got {instances.shape}")
    if sols.ndim != 2 or sols.shape[1] != instances.shape[1]:
        raise ValueError(f"Shape mismatch between sols {sols.shape} and instances {instances.shape}")

    n_instances = instances.shape[0]
    n_items = instances.shape[1]

    mismatches = []
    for idx in range(n_instances):
        wp = instances[idx]  # shape (10, 2)
        # error test
        # if idx == 0:
        #     print(idx)
        #     print(wp)
        #     wp[0][1] = 100
        #     print(wp)

        weights = wp[:, 0].astype(float)
        prices = wp[:, 1].astype(float)
        provided = sols[idx].astype(float)

        # Basic checks on provided solution
        binary_ok = is_binary_vector(provided)
        provided_rounded = np.rint(provided).astype(int)

        # Feasibility of provided solution
        provided_weight = float(np.dot(weights, provided_rounded))
        feasible = provided_weight <= capacity + 1e-9
        provided_obj = float(np.dot(prices, provided_rounded))

        # Solve with Gurobi
        x_opt, obj_opt = solve_knapsack_gurobi(weights, prices, capacity)

        # Compare objective values (robust to multiple optimal solutions)
        obj_match = abs(obj_opt - provided_obj) <= atol_obj

        if (not binary_ok) or (not feasible) or (not obj_match):
            mismatches.append(
                dict(
                    index=idx,
                    binary_ok=binary_ok,
                    feasible=feasible,
                    provided_obj=provided_obj,
                    optimal_obj=obj_opt,
                    provided_weight=provided_weight,
                    capacity=float(capacity),
                    weights=weights,
                    prices=prices,
                    provided_solution=provided_rounded,
                    optimal_solution=x_opt,
                )
            )

    # Print report
    print(f"Checked {n_instances} instances with capacity={capacity}.")
    print(f"Mismatches: {len(mismatches)}\n")
    for m in mismatches:
        print("== MISMATCH ==")
        print(f"index: {m['index']}")
        print(f"binary_ok: {m['binary_ok']}, feasible: {m['feasible']}")
        print(f"provided_obj: {m['provided_obj']:.6f}, optimal_obj: {m['optimal_obj']:.6f}")
        print(f"provided_weight: {m['provided_weight']:.6f}, capacity: {m['capacity']:.6f}")
        print(f"weights: {np.array2string(m['weights'], precision=6, separator=', ')}")
        print(f"prices: {np.array2string(m['prices'], precision=6, separator=', ')}")
        print(f"provided_solution: {m['provided_solution']}")
        print(f"optimal_solution:   {m['optimal_solution']}")
        print("")

    return len(mismatches)


def main():
    parser = argparse.ArgumentParser(description="Validate knapsack test solutions with Gurobi")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "knapsack"),
        help="Directory containing test_instances.npy and test_sols.npy",
    )
    parser.add_argument("--capacity", type=float, default=100.0, help="Knapsack capacity")
    parser.add_argument(
        "--atol-obj",
        type=float,
        default=1e-6,
        help="Absolute tolerance when comparing objective values",
    )
    args = parser.parse_args()

    mismatches = validate(args.dataset_dir, args.capacity, args.atol_obj)
    if mismatches > 0:
        exit(1)


if __name__ == "__main__":
    main()


