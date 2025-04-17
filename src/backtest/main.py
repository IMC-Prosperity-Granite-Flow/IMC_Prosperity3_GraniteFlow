from backtester import Backtester
from src.strategy.round4.round4_v02 import Trader
if __name__ == '__main__':
    backtester = Backtester(Trader)
    data_path = '/Users/IvanTang/quant/IMC_Prosperity3_GraniteFlow/data/data-bottles'
    Total_pnl, product_pnl = backtester.backtest(4, 3, data_path=data_path)
    print(f"Total PnL: {Total_pnl}, Product PnL: {product_pnl}")
