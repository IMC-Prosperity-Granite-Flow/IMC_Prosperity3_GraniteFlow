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

class MacaronStrategy(Strategy):
    def __init__(self, symbol, position_limit, conversion_limit):
        super().__init__(symbol, position_limit)
        self.conversion_limit = conversion_limit
        self.camilla_trades_history = []  # 存最近时间步的Camilla订单
        self.camilla_position = 0
        self.seen_trades = set()  # 存放已处理过的trade的唯一标识

    def process_market_data(self, state: TradingState):
        market_trades = state.market_trades.get(self.symbol, [])
        new_trades = []
        logger.print(market_trades)
        if len(market_trades) >= 1:
            for trade in market_trades:
                if trade.symbol != "MAGNIFICENT_MACARONS":
                    continue
                trade_id = (trade.timestamp, trade.price, trade.quantity, trade.buyer, trade.seller)
                if trade_id in self.seen_trades:
                    continue  # 已经处理过，跳过
                if trade.buyer == trade.seller:
                    continue
                self.seen_trades.add(trade_id)

                # 判断是否是 Camilla 的成交
                if trade.buyer == "Camilla":
                    new_trades.append(Order(self.symbol, trade.price, trade.quantity))
                    self.camilla_position += trade.quantity
                elif trade.seller == "Camilla":
                    new_trades.append(Order(self.symbol, trade.price, -trade.quantity))
                    self.camilla_position -= trade.quantity
        logger.print(new_trades)

        # 添加到历史
        self.camilla_trades_history.append(new_trades)
        if len(self.camilla_trades_history) > 10:
            self.camilla_trades_history.pop(0)

    def generate_orders(self, state: TradingState):
        self.process_market_data(state)
        orders = []
        position = state.position.get(self.symbol, 0)

        # 跟踪目标仓位为 Camilla 的当前持仓
        target_position = self.camilla_position
        delta = target_position - position

        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        buy_orders = order_depth.buy_orders.items()   # 买 -> 市场卖单
        sell_orders = order_depth.sell_orders.items() # 卖 -> 市场买单
        logger.print(f"delta: {delta}")
        # 如果我们要买入
        if delta > 0:
            quantity_to_buy = min(delta, self.position_limit - position)
            logger.print(f"buy {quantity_to_buy}")
            for bid_price, bid_volume in sell_orders:
                vol = min(quantity_to_buy, -bid_volume)

                if vol <= 0:
                    continue
                orders.append(Order(self.symbol, bid_price, vol))
                quantity_to_buy -= vol
                if quantity_to_buy <= 0:
                    break

        # 如果我们要卖出
        elif delta < 0:
            quantity_to_sell = min(-delta, self.position_limit + position)
            logger.print(f"sell {quantity_to_sell}")
            for ask_price, ask_volume in buy_orders:
                vol = min(quantity_to_sell, ask_volume)
                if vol <= 0:
                    continue
                orders.append(Order(self.symbol, ask_price, -vol))
                quantity_to_sell -= vol
                if quantity_to_sell <= 0:
                    break
        logger.print(orders)
        return orders

    def run(self, state: TradingState) -> Tuple[List[Order], dict, int]:
        orders = self.generate_orders(state)
        conversions = 0
        strategy_state = self.save_state(state)
        return orders, strategy_state, conversions
        
class Config:
    def __init__(self):
        self.PRODUCT_CONFIG = {
            "MAGNIFICENT_MACARONS": {
                "strategy_cls": MacaronStrategy,
                "symbol": "MAGNIFICENT_MACARONS",
                "position_limit": 75,
                "conversion_limit": 10
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

        # 先处理马卡龙转换请求
        if "MAGNIFICENT_MACARONS" in self.strategies and (
                "MAGNIFICENT_MACARONS" in state.order_depths or
                (state.observations and "MAGNIFICENT_MACARONS" in state.observations.conversionObservations)):

            strategy = self.strategies["MAGNIFICENT_MACARONS"]
            if "MAGNIFICENT_MACARONS" in trader_data:
                strategy.load_state(state)

            # 马卡龙策略返回额外的conversions
            macaron_orders, strategy_state, macaron_conversions = strategy.run(state)
            orders["MAGNIFICENT_MACARONS"] = macaron_orders
            new_trader_data["MAGNIFICENT_MACARONS"] = strategy_state
            conversions = macaron_conversions

        # 处理其他产品订单
        for product, strategy in self.strategies.items():
            # 跳过已处理的马卡龙
            if product == "MAGNIFICENT_MACARONS":
                continue

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


        logger.print(orders["MAGNIFICENT_MACARONS"])
        trader_data.update(new_trader_data)
        trader_data = json.dumps(trader_data)
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data