## 常用命令

### 运行实验
具体配置在
- [base.yaml](../experiments/knapsack/base.yaml)
- [comboptnet.yaml](../experiments/knapsack/comboptnet.yaml)
```
python3 main.py experiments/knapsack/comboptnet.yaml
```

knapsack问题容量, knapsack场景会使用此配置写入 KnapsackConstraintLearningTrainer
```angular2html
"data_params":
  "base_dataset_path": "./data/custom_datasets" # Add correct dataset path here ".../datasets"
  "dataset_type": "knapsack"
  "cap": 100
```

```angular2html
    if cap is not None and trainer_params.get('trainer_name') == 'KnapsackConstraintLearningTrainer':
        try:
            normalized_cap = float(cap) / 100.0
            trainer_params.setdefault('model_params', {}) \
                          .setdefault('backbone_module_params', {})['knapsack_capacity'] = normalized_cap
        except Exception:
            pass
```


### 查看数据集
具体文件在代码中配置

```
python scripts/inspect_knapsack.py \
 --dataset-dir=data/custom_datasets/knapsack \
 --limit=5
```

### 验证最优解是否正确, 文件配置在代码中
使用最优解文件和 solver 求解结果比对
具体文件在代码中配置

```angular2html
python3 scripts/validate_knapsack_gurobi.py \
--dataset-dir=data/custom_datasets/knapsack \
--capacity=100
```

```angular2html
    instances_path = os.path.join(dataset_dir, "test_instances.npy")
    sols_path = os.path.join(dataset_dir, "test_sols.npy")
```


### 生成不同容量的最优解文件

```angular2html
python scripts/generate_knapsack_solutions_gurobi.py \
--dataset-dir=data/custom_datasets/knapsack \
--capacity 150

```

### 计算后遗憾值
只允许drop, 或者允许 drop and add在文件中切换函数

python scripts/two_stage_knapsack_eval.py \
--dataset-dir data/custom_datasets_700/knapsack \
--capacity 100 \
--purchase-fee 1.0 \
--compensation-fee 5 \
--preview 5 --limit 500






 