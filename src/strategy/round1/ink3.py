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
                "max_position": 50,
                "price_zones": [
                    {"min": 500, "max": 1800, "spread": 15, "layers": 3},  # 超卖区
                    {"min": 1800, "max": 1860, "spread": 3, "layers": 5},  # 低位震荡区
                    {"min": 1860, "max": 1950, "spread": 3, "layers": 8},  # 主力交易区
                    {"min": 1950, "max": 2050, "spread": 3, "layers": 10},  # 核心波动区
                    {"min": 2050, "max": 2150, "spread": 10, "layers": 5},  # 高位震荡区
                    {"min": 2150, "max": 4000, "spread": 15, "layers": 3}  # 超买区
                ],
                "base_quantity": 5,  # 每层基础数量
                "safety_margin": 3,  # 安全边界ticks
                "trend_window": 50,
            }
        }
        self.price_history = {}
        self.price_predictions = {}
        self.ma_short = {}
        self.ma_long = {}


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
            current_value = (best_bid + best_ask) / 2 if best_bid and best_ask else 1970  # 使用平均价格作为默认值

        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.last_true_value[symbol] = current_value

        self.price_history[symbol].append(current_value)

        history_limit = 50
        if len(self.price_history[symbol]) > history_limit:
            self.price_history[symbol] = self.price_history[symbol][-history_limit:]


        return current_value


    def get_best_orders(self, symbol: str, current_value: float, order_depth: OrderDepth, position: int) -> List[Order]:
        """根据估计的真实价值和当前市场状况生成最佳订单"""
        config = self.product_config[symbol]
        orders = []

        max_position = config["max_position"]
        available_buy = max(0, max_position - position)
        available_sell = max(0, max_position + position)

        if len(self.price_history[symbol]) >= config["trend_window"]:
            self.ma_short[symbol] = np.mean(self.price_history[symbol][-config["trend_window"]:])
        else:
            self.ma_short[symbol] = current_value


        current_zone = None
        for zone in config["price_zones"]:
            if zone["min"] <= self.ma_short[symbol] < zone["max"]:
                current_zone = zone
                break
        if not current_zone:
            logger.print(f"MA50超出预设区间")
            return orders


        # 生成区间边界订单
        zone_min = current_zone["min"]
        zone_max = current_zone["max"]
        spread = current_zone["spread"]

        # 计算基准价格
        buy_base = zone_min + spread
        sell_base = zone_max - spread

        if available_buy > 0:
            orders.append(Order(symbol, buy_base, available_buy))

        if available_sell > 0:
            orders.append(Order(symbol, sell_base, -available_sell))

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

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
