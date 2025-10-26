import argparse
import os
from typing import Tuple

import numpy as np


def resolve_default_dataset_dir() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", "custom_datasets", "knapsack")


def load_data(dataset_dir: str, capacity: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    instances_path = os.path.join(dataset_dir, "test_instances.npy")
    cap_int = int(round(capacity))
    sols_path = os.path.join(dataset_dir, f"test_sols_pred_cap{cap_int}.npy")
    opt_sols_path = os.path.join(dataset_dir, f"test_sols_cap{cap_int}.npy")
    if not os.path.isfile(instances_path):
        raise FileNotFoundError(f"Missing file: {instances_path}")
    if not os.path.isfile(sols_path):
        raise FileNotFoundError(f"Missing file: {sols_path}")
    if not os.path.isfile(opt_sols_path):
        raise FileNotFoundError(f"Missing file: {opt_sols_path}")
    instances = np.load(instances_path, allow_pickle=False)  # (N, 10, 2) => [weight, price]
    sols = np.load(sols_path, allow_pickle=False)  # (N, 10) => 0/1 predicted solution
    opt_sols = np.load(opt_sols_path, allow_pickle=False)  # (N, 10) => optimal solution for this capacity
    if instances.ndim != 3 or instances.shape[2] != 2:
        raise ValueError(f"Expected instances shape (N, 10, 2). Got {instances.shape}")
    if sols.ndim != 2 or sols.shape[1] != instances.shape[1]:
        raise ValueError(f"Shape mismatch between sols {sols.shape} and instances {instances.shape}")
    if opt_sols.ndim != 2 or opt_sols.shape != sols.shape:
        raise ValueError(f"Shape mismatch between opt_sols {opt_sols.shape} and sols {sols.shape}")
    return instances, sols, opt_sols


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


def two_stage_correction_obj_only_drop(
        pred_sol: np.ndarray,
        weights: np.ndarray,
        prices: np.ndarray,
        capacity: float,
        purchase_fee: float,
        compensation_fee: float,
) -> Tuple[np.ndarray, float, int]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError(
            "Failed to import gurobipy. Ensure Gurobi is installed and licensed."
        ) from e

    n = pred_sol.shape[0]
    pred = np.rint(pred_sol).astype(int)

    model = gp.Model()
    model.Params.OutputFlag = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    sigma = model.addVars(n, vtype=GRB.BINARY, name="sigma")

    # Objective (fixed penalty per dropped item):
    obj = gp.quicksum(purchase_fee * prices[i] * x[i] - compensation_fee * prices[i] * sigma[i] for i in range(n))  # price-weighted penalty (disabled)
    # obj = (gp.quicksum(purchase_fee * prices[i] * x[i] for i in range(n)) - compensation_fee * gp.quicksum(sigma[i] for i in range(n)))
    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity after drops: (x - sigma) must satisfy capacity with real weights
    model.addConstr(gp.quicksum(weights[i] * (x[i] - sigma[i]) for i in range(n)) <= capacity, name="capacity")

    # Cannot add items not predicted; can only drop predicted ones
    for i in range(n):
        model.addConstr(x[i] == int(pred[i]), name=f"fix_pred_{i}")
        model.addConstr(x[i] >= sigma[i], name=f"sigma_le_x_{i}")

    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find optimal solution. Status: {model.status}")

    kept = np.array([int(round(x[i].X - sigma[i].X)) for i in range(n)], dtype=np.int64)
    num_dropped = int(np.sum(pred - kept))
    objective = float(model.objVal)
    return kept, objective, num_dropped


def two_stage_correction_obj_add_drop(
        pred_sol: np.ndarray,
        weights: np.ndarray,
        prices: np.ndarray,
        capacity: float,
        purchase_fee: float,
        compensation_fee: float,
) -> Tuple[np.ndarray, float, int]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError(
            "Failed to import gurobipy. Ensure Gurobi is installed and licensed."
        ) from e

    n = pred_sol.shape[0]
    pred = np.rint(pred_sol).astype(int)

    model = gp.Model()
    model.Params.OutputFlag = 0

    # Final selection variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    # Change indicators relative to prediction
    delta_plus = model.addVars(n, vtype=GRB.BINARY, name="delta_plus")  # add when pred=0
    delta_minus = model.addVars(n, vtype=GRB.BINARY, name="delta_minus")  # drop when pred=1

    # Objective: revenue from final x minus fixed penalty per change (add or drop)
    obj = (
            gp.quicksum(purchase_fee * prices[i] * x[i] for i in range(n))
            - compensation_fee * (gp.quicksum(delta_plus[i] for i in range(n)) +
                                  gp.quicksum(delta_minus[i] for i in range(n)))
    )
    model.setObjective(obj, GRB.MAXIMIZE)

    # Capacity on final selection
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity")

    # Link changes to prediction: x = pred + delta_plus - delta_minus
    for i in range(n):
        model.addConstr(x[i] == int(pred[i]) + delta_plus[i] - delta_minus[i], name=f"link_{i}")
        model.addConstr(delta_plus[i] + delta_minus[i] <= 1, name=f"no_both_changes_{i}")
        if int(pred[i]) == 1:
            model.addConstr(delta_plus[i] == 0, name=f"no_add_if_pred1_{i}")
        else:
            model.addConstr(delta_minus[i] == 0, name=f"no_drop_if_pred0_{i}")

    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find optimal solution. Status: {model.status}")

    final_x = np.array([int(round(x[i].X)) for i in range(n)], dtype=np.int64)
    num_changes = int(round(sum(delta_plus[i].X + delta_minus[i].X for i in range(n))))
    objective = float(model.objVal)
    return final_x, objective, num_changes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage knapsack correction evaluation (capacity=150)")
    parser.add_argument("--dataset-dir", type=str, default=resolve_default_dataset_dir(),
                        help="Path to knapsack dataset directory")
    parser.add_argument("--capacity", type=float, default=150.0, help="Knapsack capacity")
    parser.add_argument("--purchase-fee", type=float, default=0.2, help="Purchase fee multiplier in objective")
    parser.add_argument("--compensation-fee", type=float, default=1.0,
                        help="Penalty multiplier for dropping predicted items")
    parser.add_argument("--preview", type=int, default=3, help="Print details for first N instances")
    parser.add_argument("--atol-obj", type=float, default=1e-6,
                        help="Absolute tolerance for comparing objective values")
    parser.add_argument("--limit", type=int, default=300, help="Evaluate only the first N instances (<=0 uses all)")
    import sys

    if len(sys.argv) == 1:
        debug_args = [
            "--dataset-dir", "../data/custom_datasets_700/knapsack",
            "--capacity", "100",
            "--purchase-fee", "1",
            "--compensation-fee", "5",
            "--preview", "15",
            "--limit", "300",
        ]
        args = parser.parse_args(debug_args)
    else:
        args = parser.parse_args()

    print(f"args:{args}")

    instances, pred_sols, opt_sols_file = load_data(args.dataset_dir, args.capacity)
    print(f"instances.shape: {instances.shape}, pred_sols.shape: {pred_sols.shape}, opt_sols.shape: {opt_sols_file.shape}")

    total_instances = instances.shape[0]
    n_use = total_instances if args.limit is None or args.limit <= 0 else min(total_instances, args.limit)
    if n_use < total_instances:
        instances = instances[:n_use]
        pred_sols = pred_sols[:n_use]
        opt_sols_file = opt_sols_file[:n_use]
        print(f"Evaluating first {n_use}/{total_instances} instances")

    n_instances, n_items, _ = instances.shape

    corrected_objs = np.zeros(n_instances, dtype=np.float64)
    optimal_objs = np.zeros(n_instances, dtype=np.float64)
    pred_feasible = np.zeros(n_instances, dtype=np.bool_)
    changed_counts = np.zeros(n_instances, dtype=np.int64)
    penalties = np.zeros(n_instances, dtype=np.float64)
    obj_stage2 = np.zeros(n_instances, dtype=np.float64)
    regrets = np.zeros(n_instances, dtype=np.float64)

    mismatch_obj_count = 0
    mismatch_vec_count = 0
    perfect_pred_vec_count = 0

    for i in range(n_instances):
        wp = instances[i]
        weights = wp[:, 0].astype(float)
        prices = wp[:, 1].astype(float)
        pred = pred_sols[i].astype(float)

        # Two-stage correction (Stage 2 with fixed prediction)
        x2, corr_obj, num_changes = two_stage_correction_obj_only_drop(
            pred_sol=pred,
            weights=weights,
            prices=prices,
            capacity=args.capacity,
            purchase_fee=args.purchase_fee,
            compensation_fee=args.compensation_fee,
        )
        if float(np.dot(weights, x2)) > args.capacity + 1e-9:
            raise RuntimeError(f"WARNING: x2 solution infeasible at index {i}")

        corrected_objs[i] = corr_obj
        changed_counts[i] = num_changes
        # Components for post-hoc regret
        pred_rounded = np.rint(pred).astype(int)

        penalty_i = float((args.compensation_fee-args.purchase_fee) * np.dot(prices, pred_rounded - x2))  # price-weighted (disabled)
        # penalty_i = float((args.purchase_fee + args.compensation_fee) * np.dot(prices, pred_rounded - x2))
        # penalty_i = float(args.compensation_fee * np.sum(np.abs(pred_rounded - x2)))

        obj2_i = float(args.purchase_fee * np.dot(prices, x2))
        penalties[i] = penalty_i
        obj_stage2[i] = obj2_i

        # Feasibility of the provided prediction
        pred_feasible[i] = float(np.dot(weights, pred_rounded)) <= args.capacity + 1e-9
        # True optimal objective for reference (Gurobi)
        x_opt, opt_obj = solve_knapsack_gurobi(weights, prices, args.capacity)
        optimal_objs[i] = float(args.purchase_fee * opt_obj)

        # Reference optimal solutions from file: compare with Gurobi
        opt_vec_file = np.rint(opt_sols_file[i]).astype(int)
        # Feasibility check for file solution
        if float(np.dot(weights, opt_vec_file)) > args.capacity + 1e-9:
            raise RuntimeError(f"WARNING: reference solution infeasible at index {i}")

        opt_obj_file_raw = float(np.dot(prices, opt_vec_file))
        if abs(opt_obj_file_raw - opt_obj) > args.atol_obj:
            mismatch_obj_count += 1
        if not np.array_equal(opt_vec_file, x_opt.astype(int)):
            mismatch_vec_count += 1

        regrets[i] = penalty_i + optimal_objs[i] - obj_stage2[i]


        # Perfect prediction stats
        if np.array_equal(pred_rounded, x_opt.astype(int)):
            perfect_pred_vec_count += 1

        if i < args.preview:
            print(f"Instance {i}:")
            print(f"  capacity: {args.capacity:.6f} ")
            print(f"  weights: {weights}, ")
            print(f"  prices:{prices}")

            print(f"---------------solution------------------")
            print(f"  pred:  {pred_rounded}")
            print(f"  x2:    {x2}")
            print(f"  x_opt: {x_opt}")

            print(f"---------------regret------------------")
            print("penalty_i + optimal_objs[i] - obj_stage2[i]", penalty_i + optimal_objs[i] - obj_stage2[i])
            print("optimal_objs[i] - corrected_objs[i]", optimal_objs[i] - corrected_objs[i])
            print(f"---------------detail------------------")
            print(f"  predicted_weight: {float(np.dot(weights, pred_rounded)):.6f} | feasible: {bool(pred_feasible[i])}")
            print(f"  corrected_obj (two-stage): {corrected_objs[i]:.6f}")
            print(f"  penalty: {penalties[i]:.6f}, obj_stage2: {obj_stage2[i]:.6f}")
            print(f"  optimal_obj (scaled by purchase_fee): {optimal_objs[i]:.6f}")
            print(f"  optimal_obj_ref(file, scaled): {args.purchase_fee * opt_obj_file_raw:.6f}")
            print(f"  posthoc_regret: {regrets[i]:.6f}")
            print(f"  changed_items: {int(changed_counts[i])}")

    print("")
    print(f"Evaluated {n_instances} instances @ capacity={args.capacity:.6f}")
    print(f"Avg corrected_obj: {corrected_objs.mean():.6f}")
    print(f"Avg optimal_obj (scaled):   {optimal_objs.mean():.6f}")
    print(f"Reference vs Gurobi objective mismatches: {mismatch_obj_count}")
    print(f"Reference vs Gurobi vector mismatches:    {mismatch_vec_count}")
    print(f"Feasible prediction ratio: {pred_feasible.mean():.4f}")
    print(f"Sum changed items (when correcting): {sum(changed_counts)}")
    print(f"Avg changed items (when correcting): {changed_counts.mean():.3f}")
    print(f"Post-hoc regret: mean {regrets.mean():.6f}")
    print(f"Perfect prediction rate (vector equality): {(perfect_pred_vec_count / n_instances):.4f} ({perfect_pred_vec_count}/{n_instances})")
