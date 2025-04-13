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

    @abstractmethod
    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        """计算标的物公允价格"""
        raise NotImplementedError

    @abstractmethod
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

    def __init__(self, symbol: str, position_limit: int, alpha: float, beta):
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
        # 策略参数

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
    def __init__(self, symbol: str, position_limit: int, ma_window: int = 200,
                 max_deviation: int = 200, band_width: float = 30, take_spread: float = 10, break_step: float = 15):
        super().__init__(symbol, position_limit)

        self.timestamp = 0

        # 策略参数
        self.ma_window = ma_window
        self.max_deviation = max_deviation
        self.band_width = band_width
        self.take_spread = take_spread
        self.break_step = break_step

        # 策略历史数据
        self.fair_value_history = deque(maxlen=ma_window)
        self.fair_value_ma200_history = deque(maxlen=ma_window)
        self.current_mode = "market_making"
        self.breakout_price: Optional[float] = None
        self.prepared_reverse = False
        self.max_breakout_distance = 0
        self.ma_middle = 0
        self.breakout_times = 0
        self.needle_direction = 0  # 1:上涨突破 -1:下跌突破

    def calculate_fair_value(self, order_depth) -> float:
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

        # 返回中间价
        return (buy_avg + sell_avg) / 2

    def generate_orders(self, state) -> List[Order]:
        logger.print(f"breakout_times: {self.breakout_times}, timestamp: {self.timestamp}")
        self.timestamp += 100
        eat_pos1 = 0
        eat_pos2 = 0
        orders = []
        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        best_ask_amount = order_depth.sell_orders[best_ask]

        fair_value = self.calculate_fair_value(state.order_depths[self.symbol])
        # 保存fair_value
        self.fair_value_history.append(fair_value)
        if len(self.fair_value_history) > self.ma_window:
            self.fair_value_history.popleft()

        ma_200 = np.mean(list(self.fair_value_history)[-200:]) if len(self.fair_value_history) >= 200 else fair_value
        ma_100 = np.mean(list(self.fair_value_history)[-100:]) if len(self.fair_value_history) >= 100 else fair_value
        # 保存ma200
        self.fair_value_ma200_history.append(ma_200)
        if len(self.fair_value_ma200_history) > self.ma_window:
            self.fair_value_ma200_history.popleft()

        available_buy = 50 - current_position
        available_sell = 50 + current_position

        logger.print("Current mode: ", self.current_mode)

## ================================= 正式策略 ================================= ##
        # Strategy 0: needle
        # needle mode 时
        if (self.current_mode == "needle"):
            # 无论仓位是否满，只要价格回归优先平仓
            if fair_value >= ma_200 - 1 and current_position > 0 and self.needle_direction == -1:
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price >= ma_200 - 1:
                            quantity = min(bid_volume, current_position)
                            orders.append(Order(self.symbol, bid_price, -quantity))
                            current_position -= quantity
                            if current_position == 0: break
                return orders

            elif fair_value <= ma_200 + 1 and current_position < 0 and self.needle_direction == 1:
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                     if ask_price <= ma_200 + 1:
                            quantity = min(-ask_volume, -current_position)
                            orders.append(Order(self.symbol, ask_price, quantity))
                            current_position += quantity
                            if current_position == 0: break
                return orders

                # 重置状态
            if current_position == 0:
                self.current_mode = "market_making"
                self.needle_direction = 0
                return orders
            # 如果仓位还没满，且价格持续下跌
            elif abs(current_position) < self.position_limit:
                # 持续吃单直到满仓
                if self.needle_direction == -1:  # 下跌插针
                    for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                        if ask_price < ma_200 - 120:
                            quantity = min(-ask_volume, available_buy)
                            if quantity > 0:
                                orders.append(Order(self.symbol, ask_price, quantity))

                elif self.needle_direction == 1:  # 上涨插针
                    for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                        if bid_price > ma_200 + 120:
                            quantity = min(bid_volume, available_sell)
                            if quantity > 0:
                                orders.append(Order(self.symbol, bid_price, -quantity))
            return orders
        # 针形态检测
        else:
            # 下跌插针检测
            if best_ask < ma_200 - 80: # 已检测到针
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                    if ask_price < ma_200 - 80:  # 绝对低点
                        quantity = min(-ask_volume, available_buy)
                        if quantity > 0:
                            orders.append(Order(self.symbol, ask_price, quantity))
                            available_buy -= quantity
                self.current_mode = "needle"
                self.needle_direction = -1
                return orders

            # 上涨插针检测
            elif best_bid > ma_200 + 80: # 已检测到针
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > ma_200 + 80:  # 绝对高点
                        quantity = min(bid_volume, available_sell)
                        if quantity > 0:
                            orders.append(Order(self.symbol, bid_price, -quantity))
                            available_sell -= quantity
                self.current_mode = "needle"
                self.needle_direction = 1
                return orders

        # Strategy 1: Market mode
        if self.current_mode == "market_making":
            if len(self.fair_value_ma200_history) < 200 or abs(
                    fair_value - self.fair_value_ma200_history[-1]) <= self.band_width:
                orders = []
                # 获取当前市场数据
                order_depth = state.order_depths[self.symbol]
                current_position = state.position.get(self.symbol, 0)
                max_position = self.position_limit
                logger.print(
                    f"fair_value: {fair_value}, current_position: {current_position}, max_position: {max_position}")

                available_buy = max(0, max_position - current_position)
                available_sell = max(0, max_position + current_position)
                logger.print(f"available_buy: {available_buy}, available_sell: {available_sell}")

                # 处理卖单（asks）的限价单
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                    if ask_price < (ma_100 - self.take_spread):
                        quantity = min(-ask_volume, available_buy)
                        if quantity > 0:
                            orders.append(Order(self.symbol, ask_price, quantity))
                            available_buy -= quantity
                            eat_pos1 += quantity
                            logger.print(f"buy {quantity} at {ask_price}")

                # 处理买单（bids）的限价单
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > (ma_100 + self.take_spread):
                        quantity = min(bid_volume, available_sell)
                        if quantity > 0:
                            orders.append(Order(self.symbol, bid_price, -quantity))
                            available_sell -= quantity
                            eat_pos2 += quantity
                            logger.print(f"sell {quantity} at {bid_price}")

                # 计算挂单价格
                buy_price = math.floor(ma_100 - self.take_spread)
                sell_price = math.ceil(ma_100 + self.take_spread)

                if current_position + eat_pos1 > 0 and best_bid >ma_100:
                    quantity = min(best_bid_amount, round(0.7*available_sell))
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_bid, -quantity))
                        available_sell -= quantity

                if current_position - eat_pos2< 0 and best_ask <ma_100:
                    quantity = min(-best_ask_amount, round(0.7*available_buy))
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_ask, quantity))
                        available_sell -= quantity

                if available_buy > 0:
                    orders.append(Order(self.symbol, buy_price, available_buy))
                if available_sell > 0:
                    orders.append(Order(self.symbol, sell_price, -available_sell))

                return orders

            elif all(x != 0 for x in self.fair_value_ma200_history):
                self.breakout_price = fair_value
                self.breakout_times += 1
                # 反向吃满
                self.direction = 1 if fair_value - self.fair_value_ma200_history[-1] else -1  # 记录突破方向

                logger.print(f"Break! Breakout price: {self.breakout_price} Break direction {self.direction}")

                # 反向吃满
                if self.direction == 1:
                    # 突破是向上的，先做多
                    for price, amount in sorted(order_depth.sell_orders.items()):
                        max_amount = min(-amount, self.position_limit - current_position)
                        if max_amount > 0:
                            orders.append(Order(self.symbol, price, max_amount))
                            logger.print(f"Up break, buy {max_amount} at {price}")

                if self.direction == -1:
                    # 突破是向下的，先做空
                    for price, amount in sorted(order_depth.buy_orders.items()):
                        max_amount = min(amount, self.position_limit + current_position)
                        if max_amount > 0:
                            orders.append(Order(self.symbol, price, -max_amount))
                            logger.print(f"Down break, sell {max_amount} at {price}")

                self.current_mode = "trend_following"

        # Strategy 2: Breakout
        elif self.current_mode == "trend_following" and self.breakout_price is not None:
            distance = fair_value - self.breakout_price
            # 记录最大突破距离：
            if abs(distance) > self.max_breakout_distance + self.break_step:
                self.max_breakout_distance = abs(distance)
            self.direction = 1 if distance > 0 else -1  # 往上突破为1 往下突破为0
            position = state.position.get(self.symbol, 0)

            # 判断价格是否回归
            logger.print(
                f"Current distance: {(fair_value - self.breakout_price) * self.direction}")
            # 回归就清仓
            if (fair_value - self.breakout_price) * self.direction < 0:
                logger.print(f"Fall back! {fair_value}")
                if position != 0:
                    logger.print(f"Close position {position}")
                    if self.direction == 1:
                        # 突破是向上的，平空
                        max_amount = min(best_bid_amount, -position)
                        orders.append(Order(self.symbol, best_bid + 1, max_amount))

                    if self.direction == -1:
                        # 突破是向下的，平多
                        max_amount = min(best_ask_amount, position)
                        orders.append(Order(self.symbol, best_ask - 1, -max_amount))

                if position == 0:
                    logger.print(f"Back to market making mode")
                    # 重置突破参数
                    self.breakout_price = None
                    self.prepared_reverse = False
                    self.direction = 0
                    self.max_breakout_distance = 0
                    self.current_mode = "market_making"

            # 如果没有回归，吃回调
            else:
                # 先检查仓位有没有反向吃满，如果没有则先吃满。注意只能做一次，不然会反复反向吃满
                if position * self.direction < self.position_limit and not self.prepared_reverse:
                    logger.print(f"Preparing reverse, current position {position}, direction {self.direction}")
                    logger.print(f"{self.position_limit - position} to fill")
                    if self.direction == 1:
                        # 突破是向上的，先做多
                        for price, amount in sorted(order_depth.sell_orders.items()):
                            max_amount = min(-amount, self.position_limit - position)
                            if max_amount > 0:
                                orders.append(Order(self.symbol, price, max_amount))

                    if self.direction == -1:
                        # 突破是向下的，先做空
                        for price, amount in sorted(order_depth.buy_orders.items()):
                            max_amount = min(amount, position - self.position_limit)
                            if max_amount > 0:
                                orders.append(Order(self.symbol, price, -max_amount))

                else:
                    logger.print(f"Starting Reverse")
                    self.prepared_reverse = True  # 反向吃满了就设置为True
                    # 只有吃满了仓位才开始反转
                    target_position = -self.direction * self.position_limit
                    delta_position = target_position - position  # 还要做多少仓位才到顶

                    if delta_position != 0 and abs(distance) >= self.max_breakout_distance:  # 只有当价格突破新高(10)的时候才下单
                        res_position = self.position_limit - position if self.direction == 1 else position + self.position_limit
                        amount = min(int(abs(distance) * self.direction * delta_position / self.max_deviation),
                                     res_position)
                        # 注意amount已经包括了direction
                        if self.direction == 1:
                            orders.append(Order(self.symbol, best_ask - 1, amount))
                        if self.direction == -1:
                            orders.append(Order(self.symbol, best_bid + 1, amount))
        return orders

    def save_state(self, state):
        return {}

    def load_state(self, state):
        pass


# PICNIC_BASKET组合策略
class BasketStrategy(Strategy):
    def __init__(self, symbols: List[str], position_limits: dict,  # 移除了main_symbol参数
                 delta1_threshold: float, delta2_threshold: float,
                 time_window: int = 100):
        # 使用第一个symbol作为虚拟主产品
        super().__init__(symbols[0], position_limits[symbols[0]])

    def calculate_fair_value(self, order_depth):
        # 返回中间价
        return 0

    def quick_trade(self, symbol: str, state: TradingState, amount: int) -> Tuple[list, int]:
        """
        快速交易函数：
        给定商品名和所需数量(amount)，正数代表买入，负数代表卖出
        返回尽可能的最佳orders和剩余数量
        """
        orders = []
        return orders, amount

    def get_price_delta_basket1(self, state: TradingState) -> float:
        return 0

    def get_price_delta_basket2(self, state: TradingState) -> float:
        return 0

    def generate_orders_basket1(self, symbol: str, state: TradingState, pairing_amount1: int, pairing_amount2: int) -> \
    List[Order]:
        orders = []

        return orders

    def generate_orders_basket2(self, symbol: str, state: TradingState, pairing_amount1: int, pairing_amount2: int) -> \
    List[Order]:
        orders = []
        return orders

    def generate_orders_croissant(self, symbol: str, state: TradingState, pairing_amount1: int, pairing_amount2: int) -> \
    List[Order]:
        orders = []
        return orders

    def generate_orders_jams(self, symbol: str, state: TradingState, pairing_amount1: int, pairing_amount2: int) -> \
    List[Order]:
        orders = []
        return orders

    def generate_orders_djembes(self, symbol: str, state: TradingState, pairing_amount1: int, pairing_amount2: int) -> \
    List[Order]:
        orders = []
        return orders

    def generate_orders(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        orders = {}

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], dict]:
        orders = self.generate_orders(state)
        strategy_state = self.save_state(state)
        return orders, strategy_state

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
                "max_deviation": 200,  # 偏离标准距离（最大距离）
                "band_width": 30,  # 波动率计算的宽度
                "take_spread": 10,  # market making mode take width
                "break_step": 15,  # price range to next reverse order
            },
            "PICNIC_BASKET_GROUP": {
                "strategy_cls": BasketStrategy,
                "symbols": ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBE"],
                "position_limits": {
                    "PICNIC_BASKET1": 60,
                    "PICNIC_BASKET2": 100,
                    "CROISSANTS": 250,
                    "JAMS": 350,
                    "DJEMBE": 60
                },
                "delta1_threshold": 10,
                "delta2_threshold": 10,
                "time_window": 100
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
        product_list = [product for product, strategy in self.strategies.items()]
        for product, strategy in self.strategies.items():
            if product in trader_data or product == "PICNIC_BASKET_GROUP":
                strategy.load_state(state)
            if product in state.order_depths or product == "PICNIC_BASKET_GROUP":
                logger.print(f"Running strategy for {product}...")
                product_orders, strategy_state = strategy.run(state)
                # 处理basket订单（包括basket1, basket2, croissants, jams, djembes）
                if isinstance(product_orders, dict):
                    logger.print("Processing basket orders...")
                    for symbol, symbol_orders in product_orders.items():
                        logger.print(f"Processing orders for {symbol}...")
                        if symbol not in orders:
                            logger.print(f"Creating new order list for {symbol}...")
                            orders[symbol] = []
                        orders[symbol].extend(symbol_orders)
                        logger.print(f"Added {len(symbol_orders)} orders for {symbol}.")
                else:
                    logger.print(f"Processing {product} orders...")
                    orders[product] = product_orders

                new_trader_data[product] = strategy_state

        trader_data.update(new_trader_data)
        trader_data = json.dumps(trader_data)
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data
