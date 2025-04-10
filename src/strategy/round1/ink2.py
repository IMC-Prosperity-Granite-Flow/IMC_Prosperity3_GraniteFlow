from collections import defaultdict, deque

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict, List

import numpy as np
import json
import jsonpickle
import math


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


# 4951
class Trader:

    def __init__(self):
        # 策略路由字典（产品名: 对应的策略方法）
        self.product_config = {
            "SQUID_INK": {
                "base_quantity": 5,  # 每层基础数量
                "trend_window": 100,
                "max_position": 50,
            }
        }
        self.price_history = {}
        self.price_predictions = {}
        self.ma_short = {}
        self.ma_long = {}
        self.std_short = {}


    def calculate_current_value(self, symbol: str, order_depth: OrderDepth) -> float:
        """基于订单簿和历史数据计算估计的真实价值"""
        config = self.product_config[symbol]

        # 使用交易量加权方式整合订单簿两侧的数据
        total_volume = 0
        total_value = 0.0

        # 处理买单（按价格从高到低排序）
        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            abs_volume = abs(volume)
            total_value += price * abs_volume
            total_volume += abs_volume

        # 处理卖单（按价格从低到高排序）
        for price, volume in sorted(order_depth.sell_orders.items()):
            abs_volume = abs(volume)
            total_value += price * abs_volume
            total_volume += abs_volume

        if total_volume > 0:
            current_value = total_value / total_volume
        else:
            # 备选方案：使用中间价
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
            current_value = (best_bid + best_ask) / 2

        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(current_value)

        history_limit = 100
        if len(self.price_history[symbol]) > history_limit:
            self.price_history[symbol] = self.price_history[symbol][-history_limit:]

        return current_value


    def get_best_orders(self, symbol: str, current_value: float, order_depth: OrderDepth, position: int) -> List[Order]:
        """根据估计的真实价值和当前市场状况生成最佳订单"""
        config = self.product_config[symbol]
        orders = []

        max_position = config["max_position"]

        if len(self.price_history[symbol]) >= config["trend_window"]:
            window_data = self.price_history[symbol][-config["trend_window"]:]
            self.ma_short[symbol] = np.mean(window_data)
            self.std_short[symbol] = np.std(window_data)
        else:
            self.ma_short[symbol] = current_value
            self.std_short[symbol] = 6


        available_buy = max(0, max_position - position)
        available_sell = max(0, max_position + position)

        # 买入逻辑（价格低于MA20 - stdev）
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < self.ma_short[symbol] - 10:
                quantity = min(-vol, available_buy)
                if quantity > 0:
                    orders.append(Order(symbol, ask, quantity))
                    available_buy -= quantity

        # 卖出逻辑（价格高于MA20 + stdev）
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > self.ma_short[symbol] + 10:
                quantity = min(vol, available_sell)
                if quantity > 0:
                    orders.append(Order(symbol, bid, -quantity))
                    available_sell -= quantity

        if available_buy > 0:
            orders.append(Order(symbol, math.floor(self.ma_short[symbol] - 10), available_buy))
        if available_sell > 0:
            orders.append(Order(symbol, math.ceil(self.ma_short[symbol] + 10), -available_sell))


        return orders


    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """游戏调用的主方法"""
        result = {}
        conversions = 0
        trader_data = json.dumps({
            "price_history": {k: v[-5:] if len(v) > 5 else v for k, v in self.price_history.items()},
        })

        symbol = "SQUID_INK"

        # 检查是否有此交易对的市场数据
        if symbol in state.order_depths:
            order_depth = state.order_depths[symbol]
            position = state.position.get(symbol, 0)

            # 只有当同时存在买单和卖单时才交易
            if order_depth.buy_orders and order_depth.sell_orders:
                # 基于订单簿和历史数据计算真实价值
                current_value = self.calculate_current_value(symbol, order_depth)

                # 根据真实价值和当前市场生成订单
                orders = self.get_best_orders(symbol, current_value, order_depth, position)

                # 将订单添加到结果中
                result[symbol] = orders
                logger.print(f"Resin Orders: {orders}")

        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data
