# main.py

import pandas as pd
from typing import Dict, List
import numpy as np
import time
import json
from collections import defaultdict
from io import StringIO
# ------------------------
# 引入你已有的 datamodel (TradingState, OrderDepth, Trade, Order, Listing, Observation等)
# ------------------------
from src.strategy.datamodel import (
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
# 2) 定义一个简单策略 (Demo 策略)


class Trader:

    def run(self, state: TradingState):
        if not state.traderData or state.traderData.strip() == "":
            trader_data = {}  # 如果为空，初始化为空字典
        else:
            try:
                trader_data = json.loads(state.traderData)
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode traderData: {state.traderData}")
                trader_data = {}

        print("traderData (Loaded):", json.dumps(trader_data))
        print("Observations: " + str(state.observations))
        result = {}
        print("Current position :", state.position)
        profit_pct_limit = 0.001

        position_limit = 50
        for product in state.order_depths:
            if product == 'KELP':
                profit_pct_limit = 0.0015
            if product == 'RAINFOREST_RESIN':
                profit_pct_limit = 0.0003
            # 取出历史 fair_price
            historical_prices = trader_data.get(product, [])
            print(f'Trading {product}')

            fair_price = self.estimate_fair_price(state, product)
            historical_prices.append(fair_price)

            # 控制历史数据长度
            trader_data[product] = historical_prices[-100:]

            # 交易
            result[product] = self.trade(state, product, profit_pct_limit, position_limit, historical_prices)

        # 存回 JSON
        traderData = json.dumps(trader_data)

        # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData

    def estimate_fair_price(self, state: TradingState, product: str) -> int:
        # Estimate fair price based on market data and trader's observations

        # 用市场买卖加权均价
        order_depth = state.order_depths.get(product, OrderDepth())
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = sum(price * amount for price, amount in order_depth.buy_orders.items()) / sum(
                amount for price, amount in order_depth.buy_orders.items())
            best_ask = sum(price * amount for price, amount in order_depth.sell_orders.items()) / sum(
                amount for price, amount in order_depth.sell_orders.items())
            print('Using order depth to estimate fair price')
            fair_price = (best_bid + best_ask) / 2
            volatility = self.calculate_market_volatility(state, product)
            spread = self.calculate_spread(volatility)
            expect_bid = fair_price - spread / 2
            expect_ask = fair_price + spread / 2
            return fair_price, expect_bid, expect_ask
        else:
            if product == 'KELP':
                print('Using default price of 2025')
                return 2025
            if product == 'RAINFOREST_RESIN':
                print('Using default price of 10000')
                return 10000

    def get_best_price(self, orders: list, depth: int):
        """ 获取第 depth 层的最优价格和数量 """
        # 判断是否越界
        if not (0 <= depth < len(orders)):
            return None, None
        return orders[depth][0], orders[depth][1]

    def calculate_market_volatility(self, state: TradingState, product: str) -> float:
        # 计算市场的波动性，可以用标准差来衡量
        recent_trades = state.market_trades.get(product, [])
        if len(recent_trades) > 1:
            prices = [trade.price for trade in recent_trades]
            return np.std(prices)  # 返回价格标准差作为波动性
        return 0  # 如果交易数据不足，则返回0

    def calculate_spread(self, volatility: float) -> float:
        # 根据市场波动性计算买卖差价
        if volatility > 0:
            return volatility * 2
        return 0.01  # 默认最小差价

    def calculate_ma(self, state):
        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            trader_data = {}

        for product in state.order_depths:
            print(f'Trading {product}')

            # 计算 mid_price
            fair_price, _, _ = self.estimate_fair_price(state, product)
            if fair_price is None:
                continue

            # 存储 mid_price
            if product not in trader_data:
                trader_data[product] = []
            trader_data[product].append(fair_price)

            # 只保留最近 20 条数据（避免 traderData 过长）
            trader_data[product] = trader_data[product][-20:]

        # 更新 traderData（存储为 JSON 字符串）
        traderData = json.dumps(trader_data)
        return traderData

    def trade(self, state, product, profit_pct_limit, position_limit, historical_prices):
        order_depth: OrderDepth = state.order_depths[product]
        buy_orders = [list(order) for order in order_depth.buy_orders.items()]
        sell_orders = [list(order) for order in order_depth.sell_orders.items()]
        orders: List[Order] = []
        position = state.position.get(product, 0)
        print("Buy Order depth : " + str(len(buy_orders)) + ", Sell order depth : " + str(len(sell_orders)))

        # 从浅到深检查订单簿
        i, j = 0, 0
        fair_price, expect_bid, expect_ask = self.estimate_fair_price(state, product)
        # 暂时锁死resin的fair_price
        if product == 'RAINFOREST_RESIN':
            fair_price = 10000
        print("Fair price : " + str(fair_price))
        print("Expect bid : " + str(expect_bid))
        print("Expect ask : " + str(expect_ask))
        # 根据持仓、动量预测、订单簿不平衡度调整fair price
        alpha = 0
        beta = 0
        gamma = 0
        print(f'Position: {position}')
        momentum = self.price_momentum(historical_prices)
        print(f'Price momentum: {momentum:.2f}')
        obi = self.orderbook_imbalance(state, product)
        print(f'Orderbook imbalance: {obi:.2f}')
        fair_price = fair_price + (alpha * position + beta * momentum + gamma * obi)
        print(f'Adjusted fair price: {fair_price:.2f}, Alpha {alpha:.2f}, Beta {beta:.2f}, Gamma {gamma:.2f}')

        while i < len(sell_orders) and j < len(buy_orders):
            ask_price, ask_amount = self.get_best_price(sell_orders, i)
            bid_price, bid_amount = self.get_best_price(buy_orders, j)

            if ask_price is None or bid_price is None:
                print(f"[Warning] ask_price or bid_price is None at depth {i}, {j}")
                break  # 防止死循环

            if ask_amount == 0:
                print(f'Ask amount at depth {i} has been fully filled. Skipping')
                i += 1
                continue
            if bid_amount == 0:
                print(f'Bid amount at depth {j} has been fully filled. Skipping')
                j += 1
                continue

            if ask_price is None:
                print(f'Ask price at depth {i} is None. Skipping')
                i += 1
                continue
            if bid_price is None:
                print(f'Bid price at depth {j} is None. Skipping')
                j += 1
                continue

            print(f'depth of ask order: {i}, price: {ask_price}, amount: {ask_amount}')
            print(f'depth of bid order: {j}, price: {bid_price}, amount: {bid_amount}')

            # 主动交易
            # ask_price小于fair_price，直接买入
            if ask_price < fair_price:
                print(f'Asking price is lower than fair price, price: {ask_price}, fair_price: {fair_price}')
                # 最大可以买入的amount
                amount = min(-ask_amount, position_limit - position)
                orders.append(Order(product, ask_price, amount))
                position += amount
                i += 1
            # 如果bid_price大于fair_price，直接买入
            if bid_price > fair_price:
                print(f'Bidding price is higher than fair price, price: {bid_price}, fair_price: {fair_price}')
                # 最大可卖出（做空）的amount
                amount = min(bid_amount, position_limit + position)
                orders.append(Order(product, bid_price, -amount))
                position -= amount
                j += 1

            '''
            这一部分也可以这样做，应该更加保守？待测试
            if ask_price < expect_ask:
                print(f'Asking price too low, price: {ask_price}, expect_ask: {expect_ask}')
                i += 1
                continue
            if bid_price > expect_bid:
                print(f'Bidding price too high, price: {bid_price}, expect_bid: {expect_bid}')
                j += 1
                continue'
            '''

            # 做市
            spread = ask_price - bid_price
            # 抢单，保证成交
            ask_price -= 1
            bid_price += 1
            if spread / fair_price > profit_pct_limit:
                print(f'Spread is profitable, spread_pct: {spread / fair_price * 100:.2f}')
                amount = min(-ask_amount, bid_amount)
                if position > 0:
                    # 如果当前持仓为正，优先卖出
                    print(f'Current position is positive {position}, selling')
                    sell_amount = min(amount, position + position_limit)
                    orders.append(Order(product, ask_price, -sell_amount))
                    sell_orders[i][1] += sell_amount
                    position -= sell_amount
                    print(f'Selling {sell_amount} at {ask_price}, Current position {position}')
                elif position < 0:
                    # 如果当前持仓为负，优先买入平仓
                    print(f'Current position is negative {position}, buying to close position')
                    buy_amount = min(amount, position_limit - position)  # 仓位限制
                    orders.append(Order(product, bid_price, buy_amount))
                    buy_orders[j][1] -= buy_amount
                    position += buy_amount
                    print(f'Buying {buy_amount} at {bid_price}, Current position {position}')
                else:
                    # 如果没有持仓，同时买入和卖出
                    amount = min(amount, position_limit)  # 仓位限制
                    print('No position, executing market making')
                    orders.append(Order(product, bid_price, amount))
                    orders.append(Order(product, ask_price, -amount))
                    buy_orders[j][1] -= amount
                    sell_orders[i][1] += amount

                    print(f'Executing market making: Buy {amount} at {bid_price}, Sell {amount} at {ask_price}')

            i += 1
            j += 1
            if i >= len(sell_orders) or j >= len(buy_orders):
                break

        return orders

    def orderbook_imbalance(self, state, product: str) -> float:
        order_depth: OrderDepth = state.order_depths[product]
        buy_orders = [(price, amount) for price, amount in order_depth.buy_orders.items()]
        sell_orders = [(price, amount) for price, amount in order_depth.sell_orders.items()]

        # 计算加权的买卖总量
        buy_pressure = sum(price * amount for price, amount in buy_orders)
        sell_pressure = sum(price * amount for price, amount in sell_orders)

        total_pressure = buy_pressure + sell_pressure

        if total_pressure == 0:
            return 0  # 避免除以0

        return (buy_pressure - sell_pressure) / total_pressure

    def price_momentum(self, historical_prices: List[int]) -> float:

        return 0.0



# ------------------------
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
    log_file = "data/log/tutorial_v1.log"
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
