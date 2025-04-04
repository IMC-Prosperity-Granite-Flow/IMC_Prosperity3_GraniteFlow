# main.py

import pandas as pd
from typing import Dict, List
import numpy as np
import time
import json
from collections import defaultdict
from io import StringIO
from strategy.tutorial_v2 import Trader
# ------------------------
# 引入你已有的 datamodel (TradingState, OrderDepth, Trade, Order, Listing, Observation等)
# ------------------------
from strategy.datamodel import (
    TradingState,
    OrderDepth,
    Trade,
    Order,
    Listing,
    Observation
)

def extract_activities_log(file_path: str) -> pd.DataFrame:
    """
    从日志文件中提取 Activities log 部分并转换为 DataFrame。
    日志文件格式示例：
        Sandbox logs:
        ...（Sandbox log 内容）

        Activities log:
        <CSV 格式数据，分隔符为";">

        Trade History:
        ...（Trade log 内容）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 定位 Activities log 部分和 Trade History 部分的起始位置
    act_idx = content.find("Activities log:")
    trade_idx = content.find("Trade History:")

    if act_idx == -1:
        print("未找到 'Activities log:' 部分")
        return pd.DataFrame()

    # Activities log 部分文本在 "Activities log:" 后面，直到 "Trade History:"（如果存在）或到文件末尾
    start = act_idx + len("Activities log:")
    if trade_idx == -1:
        csv_text = content[start:]
    else:
        csv_text = content[start:trade_idx]

    # 清理提取的文本
    csv_text = csv_text.strip()
    if not csv_text:
        print("提取到的 Activities log 文本为空")
        return pd.DataFrame()

    # 用 StringIO 将文本转换为文件对象，并指定分隔符为";"
    csv_io = StringIO(csv_text)
    try:
        df = pd.read_csv(csv_io, sep=";")
    except Exception as e:
        print("解析 CSV 时发生错误:", e)
        return pd.DataFrame()

    return df




# ------------------------
# 1) 从日志文件读取活动日志并转换为 DataFrame
# ------------------------
def load_market_data_from_log(log_file_path: str) -> pd.DataFrame:
    """
    利用 extract_activities.py 中的函数提取活动日志部分，并返回 DataFrame。
    该 DataFrame 应包含至少: [timestamp, symbol, bid_price_1, bid_volume_1, ask_price_1, ask_volume_1]
    """
    df_activities = extract_activities_log(log_file_path)
    # 如果活动日志里还有更多档位 (bid_price_2, bid_volume_2 等)，可以自行扩展
    return df_activities


# ------------------------
# 2) 导入策略 (Demo 策略)

class MyStrategy:
    """
    根据日志中的 mid_price 简单地决定下单方向：
      - 如果 mid < 10000, 买1手
      - 如果 mid > 10000, 卖1手
    当然，你可以根据实际数据进行更复杂的逻辑。
    """
    def run(self, state: TradingState):
        orders = {}
        # conversions 不做额外操作，示例中返回0即可
        conversions = 0
        trader_data = state.traderData  # 如果需要持久化信息可用这个

        for symbol, od in state.order_depths.items():
            # 要求最少有 buy/sell orders 才能算出 mid
            if len(od.buy_orders) == 0 or len(od.sell_orders) == 0:
                continue
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = 0.5 * (best_bid + best_ask)

            # 简单判断 mid_price 是否在某值上下
            # 具体数值根据实际行情调节
            if mid_price < 10000:
                # 买单：限价下在 best_ask
                orders[symbol] = [Order(symbol, int(best_ask), +1)]
            elif mid_price > 10000:
                # 卖单：限价下在 best_bid
                orders[symbol] = [Order(symbol, int(best_bid), -1)]
            else:
                # 如果恰好等于10000，不操作
                orders[symbol] = []
        return orders, conversions, trader_data


# ------------------------
# 3) 简易回测器
# ------------------------
class Backtester:
    def __init__(self,
                 strategy,
                 listings: Dict[str, Listing],
                 position_limit: Dict[str, int],
                 market_data: pd.DataFrame):
        self.strategy = strategy
        self.listings = listings
        self.position_limit = position_limit
        self.market_data = market_data.sort_values(by="timestamp")
        # 初始持仓 & 现金
        self.current_position = {sym: 0 for sym in self.listings.keys()}
        self.cash = {sym: 0.0 for sym in self.listings.keys()}

        self.pnl_history = []
        self.trades = []
        self.run_times = []

    def run(self):
        # 以 timestamp 分组
        grouped = self.market_data.groupby("timestamp")
        for timestamp, group in grouped:
            order_depths = self._build_order_depths(group)
            market_trades = defaultdict(list)
            own_trades = defaultdict(list)

            # 构造一个最简单的 Observation (如果需要可放更多信息)
            observation = Observation(plainValueObservations={}, conversionObservations={})
            state = TradingState(
                traderData="",
                timestamp=timestamp,
                listings=self.listings,
                order_depths=order_depths,
                own_trades=own_trades,
                market_trades=market_trades,
                position=self.current_position,
                observations=observation
            )

            start_time = time.time()
            orders, conversions, trader_data = self.strategy.run(state)
            end_time = time.time()
            self.run_times.append(end_time - start_time)

            # 简单撮合：策略订单和对手盘匹配
            for symbol, order_list in orders.items():
                matched = self._match_orders(timestamp, symbol, order_list, order_depths[symbol])
                if matched:
                    own_trades[symbol].extend(matched)

            # 记录 PNL
            self._calc_pnl(order_depths)

        return self.trades, self.pnl_history

    def _build_order_depths(self, group: pd.DataFrame) -> Dict[str, OrderDepth]:
        order_depths = {}
        for _, row in group.iterrows():
            symbol = row["product"]
            od = OrderDepth()
            # 仅演示 1 档 bid/ask
            if not pd.isna(row.get("bid_price_1", None)):
                od.buy_orders[int(row["bid_price_1"])] = int(row["bid_volume_1"])
            if not pd.isna(row.get("bid_price_2", None)):
                od.buy_orders[int(row["bid_price_2"])] = int(row["bid_volume_2"])
            if not pd.isna(row.get("bid_price_3", None)):
                od.buy_orders[int(row["bid_price_3"])] = int(row["bid_volume_3"])

            if not pd.isna(row.get("ask_price_1", None)):
                od.sell_orders[int(row["ask_price_1"])] = int(row["ask_volume_1"]) * -1
            if not pd.isna(row.get("ask_price_2", None)):
                od.sell_orders[int(row["ask_price_2"])] = int(row["ask_volume_2"]) * -1
            if not pd.isna(row.get("ask_price_3", None)):
                od.sell_orders[int(row["ask_price_3"])] = int(row["ask_volume_3"]) * -1
                
            order_depths[symbol] = od
        return order_depths

    def _match_orders(self, timestamp: int, symbol: str, order_list: List[Order], od: OrderDepth):
        trades_done = []
        for order in order_list:
            if order.quantity == 0:
                continue
            # 买单
            if order.quantity > 0:
                # 与卖盘(ask)匹配
                sorted_asks = sorted(od.sell_orders.items(), key=lambda x: x[0])  # 价从小到大
                for ask_price, ask_volume in sorted_asks:
                    if ask_price > order.price:
                        break
                    possible_qty = min(order.quantity, abs(ask_volume))
                    # 检查限仓
                    if abs(self.current_position[symbol] + possible_qty) > self.position_limit[symbol]:
                        break
                    # 更新持仓与现金
                    self.current_position[symbol] += possible_qty
                    self.cash[symbol] -= possible_qty * ask_price
                    # 更新 order_depth
                    od.sell_orders[ask_price] += possible_qty
                    if od.sell_orders[ask_price] == 0:
                        del od.sell_orders[ask_price]
                    # 记录成交
                    trades_done.append(Trade(symbol, ask_price, possible_qty, "MY_SIDE", "MARKET", timestamp))
                    order.quantity -= possible_qty
                    if order.quantity == 0:
                        break
            else:
                # 卖单
                sorted_bids = sorted(od.buy_orders.items(), key=lambda x: x[0], reverse=True)
                for bid_price, bid_volume in sorted_bids:
                    if bid_price < order.price:
                        break
                    possible_qty = min(abs(order.quantity), bid_volume)
                    if abs(self.current_position[symbol] - possible_qty) > self.position_limit[symbol]:
                        break
                    self.current_position[symbol] -= possible_qty
                    self.cash[symbol] += possible_qty * bid_price
                    od.buy_orders[bid_price] -= possible_qty
                    if od.buy_orders[bid_price] == 0:
                        del od.buy_orders[bid_price]
                    trades_done.append(Trade(symbol, bid_price, possible_qty, "MARKET", "MY_SIDE", timestamp))
                    order.quantity += possible_qty
                    if order.quantity == 0:
                        break

        # 记录
        for t in trades_done:
            self.trades.append({
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "price": t.price,
                "quantity": t.quantity,
                "buyer": t.buyer,
                "seller": t.seller
            })

        return trades_done

    def _calc_pnl(self, order_depths: Dict[str, OrderDepth]):
        # 用买1/卖1 均价来做一个最简单的 MTM
        total_pnl = 0.0
        for sym in self.listings.keys():
            symbol_pnl = self.cash[sym]
            od = order_depths.get(sym, None)
            if od is None or (len(od.buy_orders) == 0 or len(od.sell_orders) == 0):
                # 如果没有订单深度则无法估值
                total_pnl += symbol_pnl
                continue
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = 0.5 * (best_bid + best_ask)
            symbol_pnl += self.current_position[sym] * mid_price
            total_pnl += symbol_pnl
        self.pnl_history.append(total_pnl)


# ------------------------
# 主函数：整合所有流程
# ------------------------
if __name__ == "__main__":
    # 1. 提取日志文件中的 Activities log
    log_file = "data/log/tutorial_v2.log"
    df_activities = load_market_data_from_log(log_file)
    print("提取到的活动日志 DataFrame:")
    print(df_activities.head())

    # 2. 构造 Listings 和 持仓限制
    #   注意：如果日志里有多个 symbol (如 RAINFOREST_RESIN、KELP)，要在此全部定义
    listings = {
        "RAINFOREST_RESIN": Listing("RAINFOREST_RESIN", "RAINFOREST_RESIN", "SEASHELLS"),
        "KELP": Listing("KELP", "KELP", "SEASHELLS")
    }
    position_limit = {
        "RAINFOREST_RESIN": 50,
        "KELP": 50
    }

    # 3. 初始化策略 & 回测器
    strategy = Trader()
    backtester = Backtester(
        strategy = strategy,
        listings = listings,
        position_limit = position_limit,
        market_data = df_activities
    )


    # 4. 运行回测
    trades, pnl_history = backtester.run()

    # 5. 查看结果
    print("\n=== 回测结果 ===")
    print("最终 PNL History:", pnl_history)
    print("全部成交记录:", trades)
    print("持仓:", backtester.current_position)
    print("现金余额:", backtester.cash)
