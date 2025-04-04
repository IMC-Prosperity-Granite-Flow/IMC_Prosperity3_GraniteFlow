# main.py
import io

import pandas as pd
from typing import Dict, List
import numpy as np
import time
import json
from collections import defaultdict
from io import StringIO
from backtester import  Backtester
from trader import  Trader
from utils import  _process_data_
# ------------------------
# 引入你已有的 datamodel (TradingState, OrderDepth, Trade, Order, Listing, Observation等)
# ------------------------
from datamodel import (
    TradingState,
    OrderDepth,
    Trade,
    Order,
    Listing,
    Observation
)
# ------------------------
# 主函数：整合所有流程
# ------------------------
if __name__ == "__main__":

    listings = {
        "RAINFOREST_RESIN": Listing("RAINFOREST_RESIN", "RAINFOREST_RESIN", "SEASHELLS"),
        "KELP": Listing("KELP", "KELP", "SEASHELLS")
    }
    position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50
    }

    log_file = "tutorial_v1.log"
    market_data, trade_history = _process_data_(log_file)

    fair_marks = {}
    fair_calculations = {
    }
    # 4. 初始化策略（Trader）和回测器（BackTester）
    strategy = Trader()
    backtester = Backtester(strategy, listings, position_limit, fair_calculations, market_data, trade_history, "beckTest.log")

    # 4. 运行回测
    backtester.run()
    print(backtester.current_position)
    print(backtester.cash)

