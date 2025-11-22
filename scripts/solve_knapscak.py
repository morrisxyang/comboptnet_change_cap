import gurobipy as gp
from gurobipy import GRB


# 背包的容量
capacity = 100
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
weights =  [29.0, 23.0, 15.0, 30.0, 25.0, 25.0, 26.0, 20.0, 17.0, 29.0]
# 物品的价值
values =  [29.9, 15.1, 35.7, 19.1, 19.4, 21.6, 40.7, 30.0, 14.2, 17.1]


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
