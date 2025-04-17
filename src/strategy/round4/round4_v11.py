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


# PICNIC_BASKET组合策略
class BasketStrategy(Strategy):
    def __init__(self, symbols: List[str], position_limits: dict,  # 移除了main_symbol参数
                delta1_mean: float, delta1_std: float, delta2_mean: float, delta2_std: float,
                time_window: int = 100):
        
        # 使用第一个symbol作为虚拟主产品
        super().__init__(symbols[0], position_limits[symbols[0]])
        
        self.symbols = symbols
        self.position_limits = position_limits

        self.delta1_mean = delta1_mean
        self.delta1_std = delta1_std
        self.delta2_mean = delta2_mean
        self.delta2_std = delta2_std

        self.time_window = time_window
        
        #记录所有产品的仓位历史，长度为100，利用self.position_history[symbol]取出对应仓位
        self.position_history = {symbol: [] for symbol in self.symbols}
        #记录fair_value历史
        self.fair_value_history = {symbol: [] for symbol in self.symbols}

    #——————工具函数——————

    def calculate_fair_value(self, order_depth):
        """使用买卖加权的price计算fair_value"""
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
    
    def get_available_amount(self, symbol: str, state: TradingState) -> int:
        """
        返回市场上已有市价单的总数量
        sell_amount, buy_amount（注意都为正数）
        """
        order_depth = state.order_depths[symbol]
        sell_amount = -sum(order_depth.sell_orders.values())
        buy_amount = sum(order_depth.buy_orders.values())
        return sell_amount, buy_amount
    
    def get_market_liquidity_limit(self, symbol: str, delta: np.ndarray, state: TradingState) -> np.ndarray:
        """
        对 delta 添加市场流动性约束，返回一个布尔 mask。
        正的 delta 表示买入，负的 delta 表示卖出。
        """
        sell_amount, buy_amount = self.get_available_amount(symbol, state)

        # delta 为正（买入），不能超过 buy_amount
        buy_mask = (delta <= buy_amount)
        # delta 为负（卖出），不能超过 sell_amount
        sell_mask = (-delta <= sell_amount)

        return buy_mask & sell_mask
    
    def quick_trade(self, symbol: str, state: TradingState, amount: int) -> Tuple[list, int]:
        """
        快速交易函数：
        给定商品名和所需数量(amount)，正数代表买入，负数代表卖出
        返回尽可能的最佳orders和剩余数量
        """
        orders = []
        order_depth = state.order_depths[symbol]
        position = state.position.get(symbol, 0)

        if amount > 0:
            for price, sell_amount in sorted(order_depth.sell_orders.items()):
                max_amount = min(-sell_amount, self.position_limits[symbol] - position, amount)
                if max_amount > 0:
                    orders.append(Order(symbol, price, max_amount))
                    position += max_amount
                    amount -= max_amount
                if amount == 0:
                    break

        elif amount < 0 :
            for price, buy_amount in sorted(order_depth.buy_orders.items()):
                max_amount = min(buy_amount, position + self.position_limits[symbol], -amount) #amount已经是负数，卖出
                if max_amount > 0:
                    #卖出
                    orders.append(Order(symbol, price, -max_amount))
                    position -= max_amount
                    amount += max_amount

                if amount == 0:
                    break
        
        return orders, amount

    def get_price_delta_basket1(self, state: TradingState) -> float:
        """
        返回PICNIC_BASKET1和其组分的价差：
        delta = basket1 - composition
        """
        basket1_order_depths = state.order_depths['PICNIC_BASKET1']
        croissants_order_depths = state.order_depths['CROISSANTS']
        jams_order_depths = state.order_depths['JAMS']
        djembes_order_depths = state.order_depths['DJEMBES']

        basket1_fair_value = self.calculate_fair_value(basket1_order_depths)
        croissants_fair_value = self.calculate_fair_value(croissants_order_depths)
        jams_fair_value = self.calculate_fair_value(jams_order_depths)
        djembes_fair_value = self.calculate_fair_value(djembes_order_depths)

        delta = basket1_fair_value - 6 * croissants_fair_value - 3 * jams_fair_value - 1 * djembes_fair_value
        return delta
        
    def get_price_delta_basket2(self, state: TradingState) -> float:
        """
        返回PICNIC_BASKET2和其组分的价差：
        delta = basket1 - composition
        """
        basket2_order_depths = state.order_depths['PICNIC_BASKET2']
        croissants_order_depths = state.order_depths['CROISSANTS']
        jams_order_depths = state.order_depths['JAMS']

        basket2_fair_value = self.calculate_fair_value(basket2_order_depths)
        croissants_fair_value = self.calculate_fair_value(croissants_order_depths)
        jams_fair_value = self.calculate_fair_value(jams_order_depths)

        delta = basket2_fair_value - 4 * croissants_fair_value - 2 * jams_fair_value

        return delta


    #———————下单模块——————
    def marketing_order(self, symbol: str, state: TradingState) -> List[Order]:
        """
        做市订单
        """
        orders = []
        return orders
    
    def generate_orders_basket1(self, symbol: str, state: TradingState, delta1: float, delta2: float) -> List[Order]:
        orders = []
        position = state.position.get(symbol, 0)
        order_depth = state.order_depths[symbol]
        fair_value = self.calculate_fair_value(order_depth)
        
        if delta1 != 0:
            orders = []
            scale = min(1, (delta1 - self.delta1_mean) / 3 * self.delta1_std)
        if delta2 != 0:
            orders = []

        #剩余仓位做市
        orders.extend(self.marketing_order(symbol, state))

        return orders

    def generate_orders_basket2(self, symbol: str, state: TradingState, delta1: float, delta2: float) -> List[Order]:
        orders = []
        position = state.position.get(symbol, 0)
        order_depth = state.order_depths[symbol]
        fair_value = self.calculate_fair_value(order_depth)
        return orders

    def generate_orders_croissant(self, symbol: str, state: TradingState, delta1: float, delta2: float) -> List[Order]:
        orders = []
        position = state.position.get(symbol, 0)
        order_depth = state.order_depths[symbol]
        fair_value = self.calculate_fair_value(order_depth)
        return orders
        
    def generate_orders_jams(self, symbol: str, state: TradingState, delta1: float, delta2: float) -> List[Order]:
        orders = []
        position = state.position.get(symbol, 0)
        order_depth = state.order_depths[symbol]
        fair_value = self.calculate_fair_value(order_depth)
        return orders
        
    def generate_orders_djembes(self, symbol: str, state: TradingState, delta1: float, delta2: float) -> List[Order]:
        orders = []
        position = state.position.get(symbol, 0)
        order_depth = state.order_depths[symbol]
        fair_value = self.calculate_fair_value(order_depth)
        return orders
            
    def generate_orders(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        orders = {}
        strategy_map = {
            'PICNIC_BASKET1': self.generate_orders_basket1,
            'PICNIC_BASKET2': self.generate_orders_basket2,
            'CROISSANTS': self.generate_orders_croissant,
            'JAMS': self.generate_orders_jams, 
            'DJEMBES': self.generate_orders_djembes,
        }   

        #获取两个品种的价差，计算仓位分配比例
        delta1 = self.get_price_delta_basket1(state)
        delta2 = self.get_price_delta_basket2(state)

        #价差过滤条件
        if abs(delta1 - self.delta1_mean) < self.delta1_std:
            delta1 = 0
        
        if abs(delta2 - self.delta2_mean) < self.delta2_std:
            delta2 = 0


        # 遍历处理所有相关产品
        for symbol in self.symbols:
            if symbol in state.order_depths:
                # 生成该symbol的订单...
                handler = strategy_map.get(symbol)
                orders[symbol] = handler(symbol, state, delta1, delta2)

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], dict]:
        orders = self.generate_orders(state)
        strategy_state = self.save_state(state)
        return orders, strategy_state
        
    def save_history(self, symbol: str, state: TradingState):
        """
        保存该产品的历史数据
        """
        order_depth = state.order_depths[symbol]
        position = state.position.get(symbol, 0)
        fair_value = self.calculate_fair_value(order_depth)

        self.position_history[symbol].append(position)
        self.fair_value_history[symbol].append(fair_value)

        if len(self.position_history[symbol]) > self.time_window:
            self.position_history[symbol] = self.position_history[symbol][-self.time_window:]

        if len(self.fair_value_history[symbol]) > self.time_window:
            self.fair_value_history[symbol] = self.fair_value_history[symbol][-self.time_window:]

        return
    
    def save_state(self, state):
        #对每个产品维护历史数据          
        for symbol in self.symbols:
            if symbol in state.order_depths:
                self.save_history(symbol, state)
                        
        return super().save_state(state)
        
    def load_state(self, state):
        return super().load_state(state)


class Config:
    def __init__(self):
        self.PRODUCT_CONFIG = {
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
                "delta1_mean": 48.8,
                "delta1_std": 85.1,
                "delta2_mean": 30.2,
                "delta2_std": 59.8,
                "time_window": 100
            },
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
            if product in trader_data or product == "PICNIC_BASKET_GROUP" or product == "VOLCANIC_ROCK_GROUP":
                strategy.load_state(state)
            if product in state.order_depths or product == "PICNIC_BASKET_GROUP" or product == "VOLCANIC_ROCK_GROUP":
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
