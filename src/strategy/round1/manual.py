from collections import defaultdict

# 交易汇率表
rates = {
    'SeaShells': {'Snowballs': 1.34, 'Pizza': 1.98, 'Silicon Nuggets': 0.64},
    'Snowballs': {'Pizza': 1.45, 'Silicon Nuggets': 0.52, 'SeaShells': 0.72},
    'Pizza': {'Snowballs': 0.7, 'Silicon Nuggets': 0.31, 'SeaShells': 0.48},
    'Silicon Nuggets': {'Snowballs': 1.95, 'Pizza': 3.1, 'SeaShells': 1.49}
}

# 所有币种
currencies = list(rates.keys())

# 存储最优路径
best_path = []
max_profit = 0

def dfs(path, current_currency, value, depth):
    global best_path, max_profit
    if depth > 5:
        return
    if depth > 0 and current_currency == 'SeaShells':
        if value > max_profit:
            max_profit = value
            best_path = path[:]
        return
    for next_currency, rate in rates.get(current_currency, {}).items():
        path.append(next_currency)
        dfs(path, next_currency, value * rate, depth + 1)
        path.pop()

# 从 SeaShells 开始
dfs(['SeaShells'], 'SeaShells', 1.0, 0)

# 输出结果
print("最佳路径：", " ➝ ".join(best_path))
print("最大收益（最终 SeaShells 数量）：", round(max_profit, 6))