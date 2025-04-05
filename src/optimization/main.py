import subprocess
import re
import random
import os 
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        print(f'Loaded config {config}')
    return config["parameters"]


def random_search(param_space, strategy_name, config, num_trials=10):
    best_pnl = -float('inf')
    best_params = {}

    for _ in range(num_trials):
        # 随机选择参数
        params = {key: random.choice(value) for key, value in param_space.items()}
        print(f"Testing with params: {params}")
        
        # 运行回测
        result = run_backtest(strategy_name, params, config)
        total_pnl = result["TOTAL"]

        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_params = params
            print(f'Pnl improved, current pnl {result}, current_params{best_params}')

    return best_params, best_pnl


def run_backtest(strategy_name, params, config):

    print(f"Running backtest with the following params: {params}")

    # 从配置文件加载基础参数
    kelp_params = config["KELP"]
    rainforest_resin_params = config["RAINFOREST_RESIN"]
    
    # 合并基础参数和优化参数
    # 只更新你要优化的参数
    for param, value in params.items():
        if param in rainforest_resin_params:
            rainforest_resin_params[param] = value  # 替换优化的参数

    # 生成注入的参数代码
    param_code = f"\nparams = {{'KELP': {kelp_params}, 'RAINFOREST_RESIN': {rainforest_resin_params}}}\n"
    
    # 读取策略文件并注入参数
    with open(f"{strategy_path}/{strategy_name}", 'r') as f:
        code = f.read()

    modified_code = code.replace("# PARAM_INJECTION_MARKER", param_code)
    
    temp_filename = f"temp_{strategy_name}"
    with open(f"{strategy_path}/{temp_filename}", 'w') as f:
        f.write(modified_code)

    # 执行回测
    result = subprocess.run(
        ["/opt/anaconda3/envs/quant/bin/prosperity3bt", temp_filename, str(0)],
        cwd=strategy_path,
        capture_output=True,
        text=True,
    )

    # 打印回测的标准输出和标准错误，检查是否有异常
    print(f"Standard Output: {result.stdout}")
    print(f"Standard Error: {result.stderr}")

    # 清理临时文件
    os.remove(f"{strategy_path}/{temp_filename}")

    kelp, resin, total = 0, 0, 0

    # 解析回测结果
    output = result.stdout
    kelp_profit = re.search(r"KELP: (\d+)", output)
    resin_profit = re.search(r"RAINFOREST_RESIN: ([\d,]+)", output)
    total_profit = re.search(r"Total profit: ([\d,]+)", output)
    
    if kelp_profit and resin_profit and total_profit:
        kelp = float(kelp_profit.group(1).replace(",", ""))
        resin = float(resin_profit.group(1).replace(",", ""))
        total = float(total_profit.group(1).replace(",", ""))
        print(f"KELP: {kelp}, RAINFOREST_RESIN: {resin}, Total profit: {total}")
    else:
        print("Could not extract some of the results.")
        print(output)
    
    profit_dict = {"KELP": kelp, "RAINFOREST_RESIN": resin, "TOTAL": total}

    return profit_dict


config_path = "/Users/IvanTang/quant/imc_prosperity3/src/optimization/tutorial_v3_config.json"
config = load_config(config_path)
strategy_path = "/Users/IvanTang/quant/imc_prosperity3/src/optimization"
strategy_name = "tutorial_v3_opti.py"

param_space = {
        "starfruit_make_width":[1, 5, 0.5],
        "starfruit_take_width":[0.5, 1.5, 0.1],
        "starfruit_timespan":[5, 15, 1]
}


best_params, best_pnl = random_search(param_space, strategy_name, num_trials=10, config = config)
print(f'Best params: {best_params}, Best pnl: {best_pnl}')