import gurobipy as gp
from gurobipy import GRB

data = [[26, 37.2],
        [25, 19],
        [31, 37],
        [33, 35.4],
        [20, 35.5],
        [32, 42.8],
        [23, 14.5],
        [27, 19.2],
        [24, 34.8],
        [32, 25.5]]

# 物品的重量
# weights = [10, 20, 30]
weights = [row[0] for row in data]

# 物品的价值
# values = [60, 100, 120]
values = [row[1] for row in data]


# 背包的容量
capacity = 100


if __name__ == '__main__':
    # 创建一个新的模型
    model = gp.Model("0_1_knapsack")

    # 创建变量
    x = model.addVars(len(values), vtype=GRB.BINARY, name="x")

    # 设置目标函数：最大化总价值
    model.setObjective(x.prod(values), GRB.MAXIMIZE)

    # 添加约束条件：所选物品的总重量不能超过背包容量
    model.addConstr(x.prod(weights) <= capacity, "capacity_constraint")

    # 优化模型
    model.optimize()

    # 输出结果
    if model.status == GRB.OPTIMAL:
        print("最优解已找到：", x)
        print("最大价值: ", model.objVal)
        for i in range(len(values)):
            if x[i].x > 0.5:
                print(f"选择物品 {i + 1}")
    else:
        print("未找到最优解。")
