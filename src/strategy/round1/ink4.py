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
    # 参数配置分离（每个品种独立配置）
    PRODUCT_CONFIG = {
        "RAINFOREST_RESIN": {
            "base_offset": 3,  # 基础报价偏移
            "max_position": 50,  # 最大持仓
            "eat_position1": 0,  # 固定的吃单的仓位（初始值）
            "eat_position2": 0,
            "level2spread": 8,  # spread超过这个值就用另一个offset
        },
        "KELP": {
            "max_position": 50,  # 最大持仓
            "eat_position1": 0,
            "eat_position2": 0,  # 固定的吃单的仓位（初始值）

        },
        "SQUID_INK": {
            "max_position": 50,  # 最大持仓
            "ma":100
        }
    }

    def __init__(self):
        self.strategy_router = {
            "RAINFOREST_RESIN": self.rainforestresin_strategy,
            "KELP": self.kelp_strategy,
            "SQUID_INK": self.ink_strategy
        }
        self.price_history = defaultdict(lambda: deque(maxlen=100))  # 保存100期价格数据
        self.ema_values = defaultdict(float)  # 存储最新的EMA值
        self.alpha = 2 / (200 + 1)  # EMA的alpha系数

    def calculate_true_value(self, order_depth: OrderDepth) -> float:
        """使用订单簿的买卖订单和成交量计算加权平均价格（修正版）"""
        # 处理买盘订单（buy_orders 的 volume 应为正数）
        buy_prices = np.array(list(order_depth.buy_orders.keys()))
        buy_volumes = np.array(list(order_depth.buy_orders.values()))
        sum_buy = np.sum(buy_prices * buy_volumes)
        sum_buy_vol = np.sum(buy_volumes)

        # 处理卖盘订单（sell_orders 的 volume 取绝对值）
        sell_prices = np.array(list(order_depth.sell_orders.keys()))
        sell_volumes = np.array([abs(v) for v in order_depth.sell_orders.values()])  # 关键修正点
        sum_sell = np.sum(sell_prices * sell_volumes)
        sum_sell_vol = np.sum(sell_volumes)

        # 计算买卖双方的 VWAP
        vwap_bid = sum_buy / sum_buy_vol if sum_buy_vol > 0 else None
        vwap_ask = sum_sell / sum_sell_vol if sum_sell_vol > 0 else None

        # 计算综合 VWAP（确保双方均有有效值）
        if vwap_bid is not None and vwap_ask is not None:
            vwap = (vwap_bid + vwap_ask) / 2
        else:
            vwap = 0.0  # 数据不完整时回退到 0
        logger.print(f"-----vwap: {vwap:.2f}-----")
        return vwap

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 10
        trader_data = "SAMPLE"

        # 预处理：计算所有产品的midprice并记录历史
        for product in self.PRODUCT_CONFIG:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                true_value = self.calculate_true_value(order_depth)
                self.price_history[product].append(true_value)  # 记录价格到历史
            # 策略调用
            od = state.order_depths.get(product, None)
            if not od or not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue

            if product in self.strategy_router:
                strategy = self.strategy_router[product]
                result[product] = strategy(
                    od,
                    state.position.get(product, 0),
                    product,
                )
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data


    def calculate_ema(self, product: str, new_price: float) -> float:
        """计算并更新指数移动平均"""
        if len(self.price_history[product]) == 0:
            return 0.0  # 无数据时返回0

        # 初始化阶段：前100个数据点用SMA
        if len(self.price_history[product]) < 100:
            return np.mean(self.price_history[product])

        # 正常EMA计算
        prev_ema = self.ema_values[product]
        new_ema = prev_ema * (1 - self.alpha) + new_price * self.alpha
        self.ema_values[product] = new_ema
        return new_ema

    def ink_strategy(self, order_depth: OrderDepth, current_pos: int, product: str) -> List[Order]:
        config = self.PRODUCT_CONFIG[product]
        orders = []
        max_pos = config["max_position"]
        history = self.price_history[product]

        # 等待足够数据（至少100个价格点）
        if len(history) < 100:
            logger.print(f"[{product}] Insufficient data ({len(history)}/100)")
            return orders

        # 计算EMA和波动率
        current_price = history[-1]  # 最新价格
        ema100 = self.calculate_ema(product, current_price)
        stdev = np.std(list(history))  # 基于100期数据的标准差

        # 计算volume imbalance（绝对值版本）
        buy_vol = sum(abs(v) for v in order_depth.buy_orders.values())
        sell_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        total_vol = buy_vol + sell_vol
        volume_imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0

        logger.print(f"EMA100: {ema100:.2f} STDEV: {stdev:.2f}")
        logger.print(f"Volume Imbalance: {volume_imbalance:.2f}")

        # 动态阈值调整（±1.5σ）
        upper_threshold = ema100 + 10
        lower_threshold = ema100 - 10

        available_buy = max(0, max_pos - current_pos)
        available_sell = max(0, max_pos + current_pos)

        # 买入逻辑（价格低于EMA - 1.5σ）
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < lower_threshold:
                quantity = min(-vol, available_buy)
                if quantity > 0:
                    orders.append(Order(product, ask, quantity))
                    available_buy -= quantity

        # 卖出逻辑（价格高于MA20 + stdev）
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > upper_threshold:
                quantity = min(vol, available_sell)
                if quantity > 0:
                    orders.append(Order(product, bid, -quantity))
                    available_sell -= quantity

        if available_buy > 0:
            orders.append(Order(product, math.floor(ema100 - 10), available_buy))
        if available_sell > 0:
            orders.append(Order(product, math.ceil(ema100 + 10), -available_sell))

        logger.print(f"Current Position: {current_pos}/{max_pos}")
        logger.print(f"Thresholds: [{lower_threshold:.2f}, {upper_threshold:.2f}]")
        return orders

    def rainforestresin_strategy(self, order_depth: OrderDepth, current_pos: int, product: str) -> List[Order]:
        config = self.PRODUCT_CONFIG[product]
        orders = []

        return orders


    def kelp_strategy(self, order_depth: OrderDepth, current_pos: int, product: str) -> List[Order]:
        config = self.PRODUCT_CONFIG[product]
        orders = []

        return orders