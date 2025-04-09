from Backtester import Backtester
from src.strategy.round1.round1_v2 import Trader
if __name__ == '__main__':
    backtester = Backtester(Trader)
    Total_pnl, product_pnl = backtester.backtest(1)
    print(f"Total PnL: {Total_pnl}, Product PnL: {product_pnl}")
