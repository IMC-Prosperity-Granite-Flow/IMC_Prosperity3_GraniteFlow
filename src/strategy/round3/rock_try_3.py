from abc import ABC, abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict, List, Optional, Tuple, Deque, Type
import numpy as np
import json
import jsonpickle
import math
from collections import deque

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Strategy(ABC):
    """策略抽象基类"""

    def __init__(self, symbol: str, position_limit: int):
        self.symbol = symbol
        self.position_limit = position_limit
        self.trader_data = {}

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        """计算标的物公允价格"""
        raise NotImplementedError

    def generate_orders(self, state: TradingState) -> List[Order]:
        """生成订单逻辑"""
        raise NotImplementedError

    def run(self, state: TradingState) -> Tuple[List[Order], dict]:
        """执行策略主逻辑"""

        # 状态预处理
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return [], {}

        # 生成订单
        self.orders = self.generate_orders(state)

        # 保存策略状态，用于下次加载（包括仓位、因子等历史信息）
        strategy_state = self.save_state(state)

        return self.orders, strategy_state

    def save_state(self, state) -> dict:
        """保存策略状态"""
        return {}

    def load_state(self, state: TradingState):
        """加载策略状态"""
        pass


class KelpStrategy(Strategy):
    """海带做市策略"""

    def __init__(self, symbol: str, position_limit: int, alpha: float, beta: float):
        super().__init__(symbol, position_limit)

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        return 0

    def generate_orders(self, state: TradingState) -> List[Order]:
        orders = []
        return orders

    def save_state(self, state) -> dict:
        return {}

    def load_state(self, state):
        pass


class RainforestResinStrategy(Strategy):
    """树脂动态做市策略"""

    def __init__(self, symbol: str, position_limit: int):
        super().__init__(symbol, position_limit)

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        return 0

    def generate_orders(self, state: TradingState) -> List[Order]:
        orders = []
        return orders

    def save_state(self, state) -> dict:
        return {}

    def load_state(self, state):
        pass


class SquidInkStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int, ma_window: int = 200):
        super().__init__(symbol, position_limit)

    def calculate_fair_value(self, order_depth) -> float:
        # 返回中间价
        return 0

    def generate_orders(self, state) -> List[Order]:
        orders = []

        return orders

    def save_state(self, state):
        return {}

    def load_state(self, state):
        pass


# PICNIC_BASKET组合策略
class BasketStrategy(Strategy):
    def __init__(self, symbols: List[str], position_limits: dict, std_window: int = 20,
                 delta1_threshold: float = 10, delta2_threshold: float = 10, time_window: int = 100):
        # 使用第一个symbol作为虚拟主产品
        super().__init__(symbols[0], position_limits[symbols[0]])

        self.symbols = symbols

    def calculate_fair_value(self, order_depth):
        # 返回中间价
        return 0

    def quick_trade(self, symbol: str, state: TradingState, amount: int) -> Tuple[list, int]:
        orders = []
        return orders, amount

    def get_price_delta_basket1(self, state: TradingState) -> float:
        return 0

    def get_price_delta_basket2(self, state: TradingState) -> float:
        return 0

    def calculate_basket_value(self, basket: str) -> float:
        return 0

    def calculate_std_dev(self, basket: str) -> float:
        return 0

    # ———————下单模块——————

    # basket统一订单生成函数
    def generate_basket_orders(self, state: TradingState) -> List[Order]:
        """生成订单逻辑 - 实现基础篮子套利策略"""
        orders = []
        return orders

    def generate_orders_basket1(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        return orders

    def generate_orders_basket2(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        return orders

    def generate_orders_croissant(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        return orders

    def generate_orders_jams(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        return orders

    def generate_orders_djembes(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        return orders

    def _get_max_possible_trade(self, symbol: str, state: TradingState, direction: int) -> int:
        return 0

    def generate_orders(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        # 生成订单（移除所有原有套利逻辑参数）
        orders = {}

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], dict]:
        orders = self.generate_orders(state)
        strategy_state = self.save_state(state)
        return orders, strategy_state

    def save_history(self, symbol: str, state: TradingState):
        return

    def save_state(self, state):
        return {}

    def load_state(self, state):
        pass


class VolcanicRockStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int):
        super().__init__(symbol, position_limit)

        self.symbol = symbol
        self.position_limit = position_limit
        self.active_atm = None
        self.current_mode = "Normal"
        self.active_time_value = 0
        self.rock_fair_value_history = deque(maxlen=80)
        self.ma_80 = 0
        self.time_counter = 0
        self.active_direction = 0

        self.voucher_config = {
            "VOLCANIC_ROCK_VOUCHER_9500": {
                "strike": 9500,
                "position_limit": 200,
                "lower_bound": None,
                "upper_bound": 9550,
            },
            "VOLCANIC_ROCK_VOUCHER_9750": {
                "strike": 9750,
                "position_limit": 200,
                "lower_bound": 9550,
                "upper_bound": 9800,
            },
            "VOLCANIC_ROCK_VOUCHER_10000": {
                "strike": 10000,
                "position_limit": 200,
                "lower_bound": 9800,
                "upper_bound": 10050,

            },
            "VOLCANIC_ROCK_VOUCHER_10250": {
                "strike": 10250,
                "position_limit": 200,
                "lower_bound": 10050,
                "upper_bound": 10300,

            },
            "VOLCANIC_ROCK_VOUCHER_10500": {
                "strike": 10500,
                "position_limit": 200,
                "lower_bound": 10300,
                "upper_bound": None
            },
        }

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        def weighted_avg(prices_vols, n=3):
            total_volume = 0
            price_sum = 0
            # 按价格排序（买单调降序，卖单调升序）
            sorted_orders = sorted(prices_vols.items(),
                                   key=lambda x: x[0],
                                   reverse=isinstance(prices_vols, dict))

            # 取前n档或全部可用档位
            for price, vol in sorted_orders[:n]:
                abs_vol = abs(vol)
                price_sum += price * abs_vol
                total_volume += abs_vol
            return price_sum / total_volume if total_volume > 0 else 0

        # 计算买卖方加权均价
        buy_avg = weighted_avg(order_depth.buy_orders, n=3)  # 买单簿是字典
        sell_avg = weighted_avg(order_depth.sell_orders, n=3)  # 卖单簿是字典
        if buy_avg and sell_avg:
            return (buy_avg + sell_avg) / 2
        elif buy_avg:
            return buy_avg
        elif sell_avg:
            return sell_avg

    def calculate_time_value(self, voucher_symbol: str, ma_80, voucher_price: float) -> float:
        config = self.voucher_config.get(voucher_symbol)
        return config["strike"] + voucher_price - ma_80 if config else 0.0

    def determine_ATM(self, ma_80) -> str:
        if ma_80 >= 10300:
            return "VOLCANIC_ROCK_VOUCHER_10500"
        elif ma_80 >= 10050:
            return "VOLCANIC_ROCK_VOUCHER_10250"
        elif ma_80 >= 9800:
            return "VOLCANIC_ROCK_VOUCHER_10000"
        elif ma_80 >= 9550:
            return "VOLCANIC_ROCK_VOUCHER_9750"  # 注意拼写错误需要修正
        else:
            return "VOLCANIC_ROCK_VOUCHER_9500"

    def generate_orders(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        rock_position = state.position.get("VOLCANIC_ROCK", 0)

        # 获取rock资产价格
        rock_order_depth = state.order_depths.get("VOLCANIC_ROCK", OrderDepth())
        S = self.calculate_mid_price(rock_order_depth)
        if S == 0:
            return orders

        # 添加rock_fair_value_history
        self.rock_fair_value_history.append(S)
        if len(self.rock_fair_value_history) > 80:
            self.rock_fair_value_history.popleft()
        if len(self.rock_fair_value_history) >= 80:
            self.ma_80 = np.mean(list(self.rock_fair_value_history)[-80:])
        else:
            self.ma_80 = S
            return orders

        # 确定当前ATM
        current_atm = self.determine_ATM(self.ma_80)
        atm_order_depth = state.order_depths.get(current_atm, OrderDepth())
        atm_price = self.calculate_mid_price(atm_order_depth)
        if atm_price == 0:
            return orders
        time_value = self.calculate_time_value(current_atm, self.ma_80, atm_price)

        # 如果当前是Normal模式
        if self.current_mode =="Normal":
            if time_value < 90:
                self.current_mode = "Abnormal"
                self.active_atm = current_atm
                self.active_time_value = time_value
                self.active_direction = -1
                # 卖出标的资产
                qty = (self.position_limit + rock_position) // 3
                if qty > 0 and rock_order_depth.buy_orders:
                    best_bid = min(rock_order_depth.buy_orders.keys())
                    best_bid_amount = rock_order_depth.buy_orders[best_bid]
                    sell_qty = min(best_bid_amount, qty)
                    orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -round(sell_qty))]
                # 卖出低行权价凭证
                active_strike = self.voucher_config[current_atm]["strike"]
                for symbol in self.voucher_config:
                    if self.voucher_config[symbol]["strike"] < active_strike:
                        depth = state.order_depths.get(symbol, OrderDepth())
                        if not depth.buy_orders:
                            continue
                        best_bid = min(depth.buy_orders.keys())
                        best_bid_amount = depth.buy_orders[best_bid]
                        pos_limit = self.voucher_config[symbol]["position_limit"]
                        current_pos = state.position.get(symbol, 0)
                        available_sell = pos_limit + current_pos
                        if available_sell > 0:
                            sell_qty = min(best_bid_amount, available_sell// 3)
                            if symbol not in orders:
                                orders[symbol] = []
                            orders[symbol].append(Order(symbol, best_bid, -round(sell_qty)))
                return orders
            if time_value > 110:
                self.current_mode = "Abnormal"
                self.active_atm = current_atm
                self.active_time_value = time_value
                self.active_direction = 1
                # 买入标的资产
                qty = (self.position_limit - rock_position) // 3
                if qty > 0 and rock_order_depth.sell_orders:
                    best_ask = min(rock_order_depth.sell_orders.keys())
                    best_ask_amount = rock_order_depth.sell_orders[best_ask]
                    buy_qty = min(-best_ask_amount, qty)
                    orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_ask, round(buy_qty))]
                # 买入低行权价凭证
                active_strike = self.voucher_config[current_atm]["strike"]
                for symbol in self.voucher_config:
                    if self.voucher_config[symbol]["strike"] < active_strike:
                        depth = state.order_depths.get(symbol, OrderDepth())
                        if not depth.sell_orders:
                            continue
                        best_ask = min(depth.sell_orders.keys())
                        best_ask_amount = depth.sell_orders[best_ask]
                        pos_limit = self.voucher_config[symbol]["position_limit"]  # 正确获取方式
                        current_pos = state.position.get(symbol, 0)
                        available_buy = pos_limit - current_pos
                        if available_buy > 0:
                            buy_qty = abs(min(-best_ask_amount, available_buy// 3))
                            if symbol not in orders:
                                orders[symbol] = []
                            orders[symbol].append(Order(symbol, best_ask, round(buy_qty)))
                return orders

        # 如果当前是Abnormal模式
        elif self.current_mode == "Abnormal":
            # 如果当前rock还在active_atm区间
            if self.active_atm == current_atm and self.active_direction == -1:
                atm_depth = state.order_depths.get(current_atm, OrderDepth())
                atm_price = self.calculate_mid_price(atm_depth)
                current_tv = self.calculate_time_value(current_atm, self.ma_80, atm_price)
                # 继续卖出标的
                if current_tv < self.active_time_value:
                    qty = (0.5*self.position_limit + rock_position) // 2
                    if qty > 0 and rock_order_depth.buy_orders:
                        best_bid = min(rock_order_depth.buy_orders.keys())
                        orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -round(qty))]

                    for symbol in self.voucher_config:
                        active_strike = self.voucher_config[current_atm]["strike"]
                        if self.voucher_config[symbol]["strike"] < active_strike:
                            depth = state.order_depths.get(symbol, OrderDepth())
                            if not depth.buy_orders:
                                continue
                            best_bid = min(depth.buy_orders.keys())
                            pos_limit = self.voucher_config[symbol]["position_limit"]  # 正确获取方式
                            current_pos = state.position.get(symbol, 0)
                            sell_qty = (0.5* pos_limit + current_pos) // 2
                            if sell_qty > 0:
                                if symbol not in orders:
                                    orders[symbol] = []
                                orders[symbol].append(Order(symbol, best_bid, -round(sell_qty)))
                    return orders
                # 平仓逻辑
                elif current_tv > 120:
                    # 平仓标的资产
                    total_position = rock_position  # 初始化总持仓计算
                    if rock_position < 0:
                        if rock_order_depth.sell_orders:
                            best_ask = max(rock_order_depth.sell_orders.keys())
                            best_ask_amount = rock_order_depth.sell_orders[best_ask]
                            available_buy = self.position_limit - rock_position
                            if available_buy > 0:
                                buy_qty = min(-best_ask_amount, math.floor(0.3*available_buy))
                                orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_ask, round(buy_qty))]
                    for symbol in self.voucher_config:
                        current_pos = state.position.get(symbol, 0)
                        total_position += current_pos
                        depth = state.order_depths.get(symbol, OrderDepth())
                        if not depth.sell_orders:
                            continue  # 需要处理无买单情况
                        best_ask = max(depth.sell_orders.keys())
                        best_ask_amount = depth.sell_orders[best_ask]
                        available_buy = 200 - current_pos  # 正确平仓数量
                        if symbol not in orders:
                            orders[symbol] = []
                        if available_buy > 0:
                            buy_qty = min(-best_ask_amount, math.floor(0.3*available_buy))
                            orders[symbol].append(Order(symbol, best_ask, buy_qty))
                    # 严格检查所有仓位
                    if total_position > 0:
                        self.current_mode = "Normal"
                        self.active_atm = None
                        self.time_counter = 0
                        self.active_time_value = 0
                        self.active_direction = 0
                    return orders

            if self.active_atm == current_atm and self.active_direction == 1:
                atm_depth = state.order_depths.get(current_atm, OrderDepth())
                atm_price = self.calculate_mid_price(atm_depth)
                current_tv = self.calculate_time_value(current_atm, self.ma_80, atm_price)
                # 继续买入标的
                if current_tv > self.active_time_value:
                    qty = (0.5*self.position_limit - rock_position) // 2
                    if qty > 0 and rock_order_depth.sell_orders:
                        best_ask = min(rock_order_depth.sell_orders.keys())
                        orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_ask, round(qty))]
                    for symbol in self.voucher_config:
                        active_strike = self.voucher_config[current_atm]["strike"]
                        if self.voucher_config[symbol]["strike"] < active_strike:
                            depth = state.order_depths.get(symbol, OrderDepth())
                            if not depth.sell_orders:
                                continue
                            best_ask = min(depth.sell_orders.keys())
                            pos_limit = self.voucher_config[symbol]["position_limit"]  # 正确获取方式
                            current_pos = state.position.get(symbol, 0)
                            buy_qty = (0.5* pos_limit - current_pos) // 2
                            if buy_qty > 0:
                                if symbol not in orders:
                                    orders[symbol] = []
                                orders[symbol].append(Order(symbol, best_ask, round(buy_qty)))
                    return orders
                # 平仓逻辑
                elif current_tv < 90:
                    # 平仓标的资产
                    total_position = rock_position  # 初始化总持仓计算
                    if rock_position != 0:
                        if rock_order_depth.buy_orders:
                            best_bid = max(rock_order_depth.buy_orders.keys())
                            best_bid_amount = rock_order_depth.buy_orders[best_bid]
                            available_sell = self.position_limit + rock_position
                            if available_sell > 0:
                                sell_qty = min(best_bid_amount, math.floor(0.3*available_sell))
                                orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -round(sell_qty))]
                    # 平仓所有凭证
                    for symbol in self.voucher_config:
                        current_pos = state.position.get(symbol, 0)
                        total_position += abs(current_pos)
                        if current_pos == 0:
                            continue
                        depth = state.order_depths.get(symbol, OrderDepth())
                        if not depth.buy_orders:
                            continue  # 需要处理无买单情况
                        best_bid = max(depth.buy_orders.keys())
                        best_bid_amount = depth.buy_orders[best_bid]
                        available_sell = 200 + current_pos  # 正确平仓数量
                        if symbol not in orders:
                            orders[symbol] = []
                        if available_sell > 0:
                            sell_qty = min(best_bid_amount, 0.3*available_sell)
                            orders[symbol].append(Order(symbol, best_bid, -round(sell_qty)))
                    # 严格检查所有仓位
                    if total_position < -10:
                        self.current_mode = "Normal"
                        self.active_atm = None
                        self.time_counter = 0
                        self.active_time_value = 0
                        self.active_direction = 0
                    return orders

            if self.active_atm != current_atm and self.active_atm is not None and self.active_direction == -1:
                if self.voucher_config[current_atm]["strike"] < self.voucher_config[self.active_atm]["strike"]:
                    self.active_atm = current_atm

                elif self.voucher_config[current_atm]["strike"] > self.voucher_config[self.active_atm]["strike"]:
                    self.time_counter += 1
                    total_position = rock_position if rock_position < 0 else 0  # 初始化总持仓计算
                    # 止损标的资产
                    if rock_position < 0:
                        if rock_order_depth.sell_orders:
                            best_ask = max(rock_order_depth.sell_orders.keys())
                            best_ask_amount = rock_order_depth.sell_orders[best_ask]
                            bound = self.voucher_config[current_atm]["lower_bound"]
                            available_buy = self.position_limit - rock_position
                            if available_buy > 0 and best_ask <= bound + 5:
                                buy_qty = min(-best_ask_amount, math.floor(0.5*available_buy))
                                orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_ask, round(buy_qty))]
                                # 止损所有凭证
                                for symbol in self.voucher_config:
                                    current_pos = state.position.get(symbol, 0)
                                    if current_pos < 0:
                                        total_position += current_pos
                                    if current_pos >= 0:
                                        continue
                                    depth = state.order_depths.get(symbol, OrderDepth())
                                    if not depth.sell_orders:
                                        continue  # 需要处理无买单情况
                                    best_ask = max(depth.sell_orders.keys())
                                    best_ask_amount = depth.sell_orders[best_ask]
                                    available_buy = 200 - current_pos  # 正确平仓数量
                                    if symbol not in orders:
                                        orders[symbol] = []
                                    if available_buy > 0:
                                        buy_qty = min(-best_ask_amount, 0.5*available_buy)
                                        orders[symbol].append(Order(symbol, best_ask, round(buy_qty)))

                    # 严格检查所有仓位
                    if total_position >= -20 or self.time_counter == 30:
                        self.current_mode = "Normal"
                        self.active_atm = None
                        self.active_time_value = 0
                        self.time_counter = 0
                        self.active_direction = 0
                    return orders

            if self.active_atm != current_atm and self.active_atm is not None and self.active_direction == 1:
                if self.voucher_config[current_atm]["strike"] > self.voucher_config[self.active_atm]["strike"]:
                    self.active_atm = current_atm

                elif self.voucher_config[current_atm]["strike"] < self.voucher_config[self.active_atm]["strike"]:
                    self.time_counter += 1
                    total_position = rock_position if rock_position > 0 else 0  # 初始化总持仓计算
                    # 止损标的资产
                    if rock_position > 0:
                        if rock_order_depth.buy_orders:
                            best_bid = max(rock_order_depth.buy_orders.keys())
                            best_bid_amount = rock_order_depth.buy_orders[best_bid]
                            bound = self.voucher_config[current_atm]["upper_bound"]
                            available_sell = self.position_limit + rock_position
                            if available_sell > 0 and best_bid >= bound - 5:
                                sell_qty = min(best_bid_amount, math.floor(0.5*available_sell))
                                orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -round(sell_qty))]
                                # 止损所有凭证
                                for symbol in self.voucher_config:
                                    current_pos = state.position.get(symbol, 0)
                                    if current_pos > 0:
                                        total_position += current_pos
                                    if current_pos <= 0:
                                        continue
                                    depth = state.order_depths.get(symbol, OrderDepth())
                                    if not depth.buy_orders:
                                        continue  # 需要处理无买单情况
                                    best_bid = max(depth.buy_orders.keys())
                                    best_bid_amount = depth.buy_orders[best_bid]
                                    available_sell = 200 + current_pos  # 正确平仓数量
                                    if symbol not in orders:
                                        orders[symbol] = []
                                    if available_sell > 0:
                                        sell_qty = min(best_bid_amount, 0.5*available_sell)
                                        orders[symbol].append(Order(symbol, best_bid, -round(sell_qty)))

                    # 严格检查所有仓位
                    if total_position < 20 or self.time_counter == 20:
                        self.current_mode = "Normal"
                        self.active_atm = None
                        self.active_time_value = 0
                        self.time_counter = 0
                        self.active_direction = 0

                    return orders


        return orders


    def save_state(self, state):
        return {}

    def load_state(self, state):
        pass


class Config:
    def __init__(self):
        self.PRODUCT_CONFIG = {
            "KELP": {
                "strategy_cls": KelpStrategy,
                "symbol": "KELP",
                "position_limit": 50,
                "alpha": 0,
                "beta": 0
            },
            "RAINFOREST_RESIN": {
                "strategy_cls": RainforestResinStrategy,
                "symbol": "RAINFOREST_RESIN",
                "position_limit": 50,  # 最大持仓
            },
            "SQUID_INK": {
                "strategy_cls": SquidInkStrategy,
                "symbol": "SQUID_INK",
                "position_limit": 50,  # 最大持仓量
                "ma_window": 200,  # 计算均价的时长
            },
            "PICNIC_BASKET_GROUP": {
                "strategy_cls": BasketStrategy,
                "symbols": ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"],
                "position_limits": {
                    "PICNIC_BASKET1": 60,
                    "PICNIC_BASKET2": 100,
                    "CROISSANTS": 250,
                    "JAMS": 350,
                    "DJEMBES": 60
                },
                "std_window": 20,
                "delta1_threshold": 10,
                "delta2_threshold": 10,
                "time_window": 100
            },
            "VOLCANIC_ROCK": {
                "strategy_cls": VolcanicRockStrategy,
                "symbol": "VOLCANIC_ROCK",
                "position_limit": 400
            }
        }


class Trader:
    def __init__(self, product_config=None):
        # 使用默认 config，或外部传入 config
        self.PRODUCT_CONFIG = product_config if product_config is not None else Config().PRODUCT_CONFIG
        self.strategies = {}
        self._init_strategies()

    def _init_strategies(self):
        for product, config in self.PRODUCT_CONFIG.items():
            if product == "PICNIC_BASKET_GROUP":
                cls = config["strategy_cls"]
                args = {k: v for k, v in config.items() if k != "strategy_cls" and k != "symbols"}
                self.strategies[product] = cls(symbols=config["symbols"], **args)

            else:
                # 常规产品初始化保持不变
                cls = config["strategy_cls"]
                args = {k: v for k, v in config.items() if k != "strategy_cls"}
                self.strategies[product] = cls(**args)

    def run(self, state: TradingState):
        conversions = 0
        # 加载历史状态
        trader_data = json.loads(state.traderData) if state.traderData else {}

        orders = {}
        new_trader_data = {}
        for product, strategy in self.strategies.items():
            if product in trader_data or product == "PICNIC_BASKET_GROUP":
                strategy.load_state(state)
            if product in state.order_depths or product == "PICNIC_BASKET_GROUP":
                product_orders, strategy_state = strategy.run(state)
                # 处理group订单
                if isinstance(product_orders, dict):
                    for symbol, symbol_orders in product_orders.items():
                        if symbol not in orders:
                            orders[symbol] = []
                        orders[symbol].extend(symbol_orders)
                else:
                    orders[product] = product_orders

                new_trader_data[product] = strategy_state

        trader_data.update(new_trader_data)
        trader_data = json.dumps(trader_data)
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data
