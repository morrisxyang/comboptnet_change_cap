## 常用命令

### 运行实验
```
python3 main.py experiments/knapsack/comboptnet.yaml
```

### 查看数据集

```
python scripts/inspect_knapsack.py \
 --dataset-dir=data/custom_datasets/knapsack \
 --limit=5
```

### 验证最优解是否正确, 文件配置在代码中
```angular2html
python3 scripts/validate_knapsack_gurobi.py \
--dataset-dir=data/custom_datasets/knapsack \
--capacity=100
```

```angular2html
    instances_path = os.path.join(dataset_dir, "test_instances.npy")
    sols_path = os.path.join(dataset_dir, "test_sols.npy")
```


### 生成不同容量的最优解

```angular2html
python scripts/generate_knapsack_solutions_gurobi.py \
--dataset-dir=data/custom_datasets/knapsack \
--capacity 150

```






knapsack问题, 容量在哪里设置





 