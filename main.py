import os
import sys
import ray
from os import path, makedirs
from data import load_dataset
from trainer import get_trainer
from utils.utils import print_eval_acc, print_train_acc, load_with_default_yaml, save_dict_as_one_line_csv
from datetime import datetime


def main(working_dir, seed, train_epochs, eval_every, use_ray, ray_params, data_params, trainer_params):
    if use_ray:
        ray.init(**ray_params)


    # If capacity is provided in data_params and we're using the knapsack trainer,
    # set backbone knapsack_capacity accordingly (scale by 1/100).
    cap = data_params.get('cap', None)
    # Append seconds-level timestamp and cap (if provided) to working_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cap is not None:
        working_dir = path.join(working_dir, f"{timestamp}_cap{cap}")
    else:
        working_dir = path.join(working_dir, f"{timestamp}")
    makedirs(working_dir, exist_ok=True)
    print(f"train_epochs: {train_epochs}, working_dir: {working_dir},")

    # Log selected hyperparameters
    try:
        bs = data_params.get('loader_params', {}).get('batch_size', None)
    except Exception:
        bs = None
    try:
        lr = trainer_params.get('optimizer_params', {}).get('lr', None)
    except Exception:
        lr = None
    print(f"batch_size: {bs}, lr: {lr}")

    if cap is not None and trainer_params.get('trainer_name') == 'KnapsackConstraintLearningTrainer':
        try:
            normalized_cap = float(cap) / 100.0
            trainer_params.setdefault('model_params', {}) \
                          .setdefault('backbone_module_params', {})['knapsack_capacity'] = normalized_cap
        except Exception:
            pass

    (train_iterator, test_iterator), metadata = load_dataset(**data_params)
    trainer = get_trainer(seed=seed, train_iterator=train_iterator, test_iterator=test_iterator, metadata=metadata,
                          **trainer_params)

    eval_metrics = trainer.evaluate()
    print_eval_acc(eval_metrics)

    for i in range(train_epochs):
        train_metrics = trainer.train_epoch()
        print_train_acc(train_metrics, epoch=i)
        if eval_every is not None and (i + 1) % eval_every == 0:
            eval_metrics = trainer.evaluate()
            print_eval_acc(eval_metrics)

    # Save test predictions at final evaluation
    import numpy as np  # retained import if needed elsewhere
    save_path = path.join(working_dir, f"test_sols_pred_cap{cap}.npy")
    eval_metrics = trainer.evaluate(save_predictions_path=save_path)
    print_eval_acc(eval_metrics)

    if use_ray:
        ray.shutdown()
    metrics = dict(**train_metrics, **eval_metrics)
    save_dict_as_one_line_csv(metrics, filename=os.path.join(working_dir, "metrics.csv"))
    return metrics


if __name__ == "__main__":
    param_path = sys.argv[1]
    param_dict = load_with_default_yaml(path=param_path)
    main(**param_dict)
