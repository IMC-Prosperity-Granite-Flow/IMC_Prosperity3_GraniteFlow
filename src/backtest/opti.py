from Backtester import Backtester
from src.strategy.round1.round1_v2 import Trader
from src.strategy.round1.round1_v2 import Config
import numpy as np



class CustomTrader(Trader):
    def __init__(self, product_params: dict[str, dict[str, float]]):
        base_config = Config().PRODUCT_CONFIG
        product_config = {k: v.copy() for k, v in base_config.items()}
        for product, params in product_params.items():
            if product in product_config:
                product_config[product].update(params)

        super().__init__(product_config=product_config)

'''
class CustomTrader(BaseTrader):
    def __init__(self, product_params):
        # 创建新配置
        new_config = {k: v.copy() for k, v in BaseTrader.PRODUCT_CONFIG.items()}
        
        # 更新参数
        for product, params in product_params.items():
            if product in new_config:
                new_config[product].update(params)
        
        # 🚨 关键步骤：覆盖类属性为实例属性
        self.PRODUCT_CONFIG = new_config  # ✅ 必须在super()之前
        
        # 最后调用父类初始化
        super().__init__()  # ✅ 此时父类__init__会使用新的PRODUCT_CONFIG
'''
def evaluate_strategy(product_params: dict[str, dict[str, float]]) -> float:
    trader = CustomTrader(product_params)
    backtester = Backtester(trader)
    total_pnl, _ = backtester.backtest(round_num=1)
    return total_pnl

if __name__ == '__main__':
    best_pnl = float('-inf')
    best_config = None

    # 以 SQUID_INK 为例，搜索两个参数
    for reversal_threshold in range(10, 30, 5):
        for price_momentum_factor in np.linspace(0.05, 0.2, 4):
            param_config = {
                "SQUID_INK": {
                    "reversal_threshold": reversal_threshold,
                    "price_momentum_factor": price_momentum_factor,
                }
            }

            pnl = evaluate_strategy(param_config)
            print(f"Params: {param_config['SQUID_INK']}, PnL: {pnl}")

            if pnl > best_pnl:
                best_pnl = pnl
                best_config = param_config

    print("Best Config:", best_config)
    print("Best PnL:", best_pnl)
