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
            "take_position1": 0,  # 固定的吃单的仓位（初始值）
            "take_position2": 0,
            "level2spread": 8,  # spread超过这个值就用另一个offset
        },
        "KELP": {
            "max_position": 50,  # 最大持仓
            "take_position1": 0,
            "take_position2": 0,  # 固定的吃单的仓位（初始值）
        }
    }

    def __init__(self):
        # 策略路由字典（产品名: 对应的策略方法）
        self.strategy_router = {
            "RAINFOREST_RESIN": self.rainforestresin_strategy,
            "KELP": self.kelp_strategy
        }

    def calculate_true_value(self, order_depth: OrderDepth) -> float:
        """基于订单簿前三档的加权中间价计算"""

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

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 10
        trader_data = "SAMPLE"

        for product in ["RAINFOREST_RESIN", "KELP"]:

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

    def rainforestresin_strategy(self, order_depth: OrderDepth, current_pos: int, product: str) -> List[Order]:
        config = self.PRODUCT_CONFIG[product]
        orders = []
        max_pos = config["max_position"]
        offset = config["base_offset"]
        eat_pos1 = config["take_position1"]
        eat_pos2 = config["take_position2"]
        level2spread = config["level2spread"]
        FIXED_MID = 10000  # 固定中间价
        available_buy = max(0, max_pos - current_pos)
        available_sell = max(0, max_pos + current_pos)
        sorted_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(order_depth.sell_orders.keys())
        best_bid = sorted_bids[0] if sorted_bids else FIXED_MID
        best_ask = sorted_asks[0] if sorted_asks else FIXED_MID
        second_bid = sorted_bids[1] if len(sorted_bids) >= 2 else best_bid
        second_ask = sorted_asks[1] if len(sorted_asks) >= 2 else best_ask

        # 吃单逻辑 ================================================
        # 处理所有低于10000的卖单（按价格升序排列）
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < FIXED_MID:
                # 计算最大可买量
                buyable = min(-vol, max_pos - current_pos)
                if buyable > 0:
                    orders.append(Order(product, ask, buyable))
                    eat_pos1 += buyable
            elif ask == FIXED_MID and available_sell< 30:
                buyable = min(-vol, max_pos - current_pos)
                orders.append(Order(product, ask, buyable))
                eat_pos1 += buyable
            else:
                break  # 后续价格更高，不再处理

        # 处理所有高于10000的买单（按价格降序排列）
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > FIXED_MID:
                # 计算最大可卖量
                sellable = min(vol, max_pos + current_pos)
                if sellable > 0:
                    orders.append(Order(product, bid, -sellable))
                    eat_pos2 += sellable

            elif bid == FIXED_MID and available_buy < 30:
                sellable = min(vol, max_pos + current_pos)
                orders.append(Order(product, bid, -sellable))
                eat_pos2 += sellable

            else:
                break  # 后续价格更低，不再处理


        # 挂单逻辑 ================================================
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread = best_ask - best_bid
        if spread > level2spread:
            offset = 4
        desired_bid = best_bid + 1
        if desired_bid>10000:
            desired_bid = second_bid +1

        desired_ask = best_ask - 1
        if desired_ask<10000:
            desired_ask = second_ask -1

        # 计算可用挂单量
        desired_buy = available_buy - eat_pos1
        desired_sell = available_sell - eat_pos2  # 固定吃单额度

        # 买盘挂单（正数表示买入）
        if desired_buy > 0 :
            orders.append(Order(product, desired_bid, desired_buy))

        # 卖盘挂单（负数表示卖出）
        if desired_sell > 0 :
            orders.append(Order(product, desired_ask, -desired_sell))

        return orders


    def kelp_strategy(self, order_depth: OrderDepth, current_pos: int, product: str) -> List[Order]:
        config = self.PRODUCT_CONFIG[product]
        orders = []
        max_pos = config["max_position"]
        eat_pos1 = config["take_position1"]
        eat_pos2 = config["take_position2"]
        available_buy = max(0, max_pos - current_pos)
        available_sell = max(0, max_pos + current_pos)

        true_value = self.calculate_true_value(order_depth)
        # 吃单逻辑
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < true_value:
                # 计算最大可买量
                buyable = min(-vol, max_pos - current_pos)
                if buyable > 0:
                    orders.append(Order(product, ask, buyable))
                    eat_pos1 += buyable
            else:
                break  # 后续价格更高，不再处理

        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > true_value:
                # 计算最大可卖量
                sellable = min(vol, max_pos + current_pos)
                if sellable > 0:
                    orders.append(Order(product, bid, -sellable))
                    eat_pos2 += sellable
            else:
                break  # 后续价格更低，不再处理

        # 挂单逻辑
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        desired_bid = best_bid + 1
        if desired_bid >= true_value:
            desired_bid = round(true_value) - 1

        desired_ask = best_ask - 1
        if desired_ask <= true_value:
            desired_ask = round(true_value) + 1

        desired_buy = available_buy - eat_pos1
        desired_sell = available_sell - eat_pos2

        if desired_buy > 0:
            orders.append(Order(product, desired_bid, desired_buy))
        if desired_sell > 0:
            orders.append(Order(product, desired_ask, -desired_sell))

        return orders