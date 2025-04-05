import csv
from src.strategy.datamodel import OrderDepth, TradingState, Order, Trade, Observation
#src.strategy.your_strategy_file import YourStrategyClass
from src.optimization.tutorial_v3_opti import Trader

from src.backtest.prosperity3bt.models import TradeMatchingMode
from prosperity3bt.runner import run_backtest
from src.backtest.prosperity3bt.file_reader import FileReader, FileSystemReader, PackageResourcesReader


trader = Trader()
result = run_backtest(trader, FileReader, 0, -2, True, 'all', )
print(result)