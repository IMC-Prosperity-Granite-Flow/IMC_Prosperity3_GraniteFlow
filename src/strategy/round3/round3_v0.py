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
        # 添加海带策略特有参数
        self.alpha = alpha  # adjusted fair price清仓系数
        self.alpha = beta  # adjusted fair price订单簿不平衡度系数
        self.trader_data = {}
        self.position_history = []

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:

        total_volume = 0
        total_value = 0.0

        # 合并处理所有买单（买单价从高到低）
        for price, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            abs_vol = abs(vol)
            total_value += price * abs_vol
            total_volume += abs_vol

        # 合并处理所有卖单（卖单价从低到高）
        for price, vol in sorted(order_depth.sell_orders.items()):
            if abs(vol) >= 10:  # 过滤掉数量小于10的卖单
                abs_vol = abs(vol)
                total_value += price * abs_vol
                total_volume += abs_vol

        if total_volume > 0:
            return total_value / total_volume
        else:
            # 如果没有订单，返回买卖中间价（兜底逻辑）
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
            return (best_bid + best_ask) / 2 if best_bid and best_ask else 0

    def generate_orders(self, state: TradingState) -> List[Order]:
        take_position1 = 0
        take_position2 = 0
        current_position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]
        fair_value = self.calculate_fair_value(order_depth)

        available_buy = max(0, self.position_limit - current_position)
        available_sell = max(0, self.position_limit + current_position)

        fair_value = fair_value - 0.03 * current_position

        orders = []

        # 吃单逻辑
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < fair_value:
                # 计算最大可买量
                buyable = min(-vol, self.position_limit - current_position)
                if buyable > 0:
                    orders.append(Order(self.symbol, ask, buyable))
                    take_position1 += buyable
            else:
                break  # 后续价格更高，不再处理

        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > fair_value:
                # 计算最大可卖量
                sellable = min(vol, self.position_limit + current_position)
                if sellable > 0:
                    orders.append(Order(self.symbol, bid, -sellable))
                    take_position2 += sellable
            else:
                break  # 后续价格更低，不再处理

        # 挂单逻辑
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0

        # 根据订单簿深度调整挂单策略
        bid_volume = sum(abs(v) for v in order_depth.buy_orders.values())
        ask_volume = sum(abs(v) for v in order_depth.sell_orders.values())
        total_volume = bid_volume + ask_volume
        volume_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

        # 考虑订单簿不平衡度来调整挂单价格
        bid_adjust = max(0, round(volume_imbalance * 2))  # 买盘压力大时降低买价
        ask_adjust = max(0, round(-volume_imbalance * 2))  # 卖盘压力大时提高卖价

        desired_bid = best_bid + 1 - bid_adjust
        if desired_bid >= fair_value:
            desired_bid = math.floor(fair_value)

        desired_ask = best_ask - 1 + ask_adjust
        if desired_ask <= fair_value:
            desired_ask = math.ceil(fair_value)

        # 根据持仓和订单簿力量调整价格
        if current_position > 25 and volume_imbalance < 0:
            # 持仓多且卖方力量大，更积极卖出
            desired_ask -= 1  # 再降1个tick
            if desired_ask <= fair_value:
                desired_ask = math.ceil(fair_value)  # 保持最低价格保护

        if current_position < -25 and volume_imbalance > 0:
            # 持仓空且买方力量大，更积极买入
            desired_bid += 1  # 再提高1个tick
            if desired_bid >= fair_value:
                desired_bid = math.floor(fair_value)  # 保持最高价格保护

        # 根据持仓和方向调整挂单量
        desired_buy = min(15, available_buy - take_position1)
        desired_sell = min(15, available_sell - take_position2)

        if desired_buy > 0:
            orders.append(Order(self.symbol, desired_bid, desired_buy))
        if desired_sell > 0:
            orders.append(Order(self.symbol, desired_ask, -desired_sell))

        return orders

    def save_state(self, state) -> dict:
        return {}

    def load_state(self, state):
        return self.position_history


class RainforestResinStrategy(Strategy):
    """树脂动态做市策略"""

    def __init__(self, symbol: str, position_limit: int):
        super().__init__(symbol, position_limit)

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        return 0

    def generate_orders(self, state: TradingState) -> List[Order]:
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        orders = []
        take_position1 = 0
        take_position2 = 0
        FIXED_MID = 10000  # 固定中间价
        available_buy = max(0, self.position_limit - position)
        available_sell = max(0, self.position_limit + position)
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
                buyable = min(-vol, self.position_limit - position)
                if buyable > 0:
                    orders.append(Order(self.symbol, ask, buyable))
                    take_position1 += buyable
            elif ask == FIXED_MID and available_sell < 30:
                buyable = min(-vol, self.position_limit - position)
                orders.append(Order(self.symbol, ask, buyable))
                take_position1 += buyable
            else:
                break  # 后续价格更高，不再处理

        # 处理所有高于10000的买单（按价格降序排列）
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > FIXED_MID:
                # 计算最大可卖量
                sellable = min(vol, self.position_limit + position)
                if sellable > 0:
                    orders.append(Order(self.symbol, bid, -sellable))
                    take_position2 += sellable

            elif bid == FIXED_MID and available_buy < 30:
                sellable = min(vol, self.position_limit + position)
                orders.append(Order(self.symbol, bid, -sellable))
                take_position2 += sellable

            else:
                break  # 后续价格更低，不再处理

        # 挂单逻辑 ================================================
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        desired_bid = best_bid + 1
        if desired_bid > 10000:
            desired_bid = second_bid + 1

        desired_ask = best_ask - 1
        if desired_ask < 10000:
            desired_ask = second_ask - 1

        # 计算可用挂单量
        desired_buy = available_buy - take_position1
        desired_sell = available_sell - take_position2  # 固定吃单额度

        # 买盘挂单（正数表示买入）
        if desired_buy > 0:
            orders.append(Order(self.symbol, desired_bid, desired_buy))

        # 卖盘挂单（负数表示卖出）
        if desired_sell > 0:
            orders.append(Order(self.symbol, desired_ask, -desired_sell))

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
                #logger.print(f"fair_value: {fair_value}, current_position: {current_position}, max_position: {max_position}")

                available_buy = max(0, max_position - current_position)
                available_sell = max(0, max_position + current_position)
                #logger.print(f"available_buy: {available_buy}, available_sell: {available_sell}")

                # 处理卖单（asks）的限价单
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                    if ask_price < (ma_100 - self.take_spread):
                        quantity = min(-ask_volume, available_buy)
                        if quantity > 0:
                            orders.append(Order(self.symbol, ask_price, quantity))
                            available_buy -= quantity
                            eat_pos1 += quantity
                            #logger.print(f"buy {quantity} at {ask_price}")

                # 处理买单（bids）的限价单
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > (ma_100 + self.take_spread):
                        quantity = min(bid_volume, available_sell)
                        if quantity > 0:
                            orders.append(Order(self.symbol, bid_price, -quantity))
                            available_sell -= quantity
                            eat_pos2 += quantity
                            #logger.print(f"sell {quantity} at {bid_price}")

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
                            #logger.print(f"Up break, buy {max_amount} at {price}")

                if self.direction == -1:
                    # 突破是向下的，先做空
                    for price, amount in sorted(order_depth.buy_orders.items()):
                        max_amount = min(amount, self.position_limit + current_position)
                        if max_amount > 0:
                            orders.append(Order(self.symbol, price, -max_amount))
                            #logger.print(f"Down break, sell {max_amount} at {price}")

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
            ##logger.print(f"Current distance: {(fair_value - self.breakout_price) * self.direction}")
            # 回归就清仓
            if (fair_value - self.breakout_price) * self.direction < 0:
                #logger.print(f"Fall back! {fair_value}")
                if position != 0:
                    #logger.print(f"Close position {position}")
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
                    #logger.print(f"{self.position_limit - position} to fill")
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
    def __init__(self, symbols: List[str], position_limits: dict,std_window: int = 20,
        delta1_threshold: float=10, delta2_threshold: float=10, time_window: int = 100):
        # 使用第一个symbol作为虚拟主产品
        super().__init__(symbols[0], position_limits[symbols[0]])

        self.symbols = symbols
        self.position_limits = position_limits

        self.delta1_history = deque(maxlen=time_window)
        self.delta2_history = deque(maxlen=time_window)
        self.std_window = std_window
        self.delta1_threshold = delta1_threshold
        self.delta2_threshold = delta2_threshold
        self.time_window = time_window
        self.std_multiplier = 2

        # 记录所有产品的仓位历史，长度为100，利用self.position_history[symbol]取出对应仓位
        self.position_history = {symbol: [] for symbol in self.symbols}
        # 记录fair_value历史
        self.fair_value_history = {symbol: [] for symbol in self.symbols}


        self.basket_composition = {
            'PICNIC_BASKET1': {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBES': 1},
            'PICNIC_BASKET2': {'CROISSANTS': 4, 'JAMS': 2}
        }
        # 存储价差历史，而不是价格历史
        self.price_diff_history = {
            'PICNIC_BASKET1': deque(maxlen=20),
            'PICNIC_BASKET2': deque(maxlen=20)
        }
        
        # 标准差缓存
        self.std_devs = {
            'PICNIC_BASKET1': 0,
            'PICNIC_BASKET2': 0
        }
        
        # 组件价格缓存
        self.component_prices = {}
        
        # 篮子特定交易参数（标准差倍数）- PB1使用更保守的参数
        self.std_thresholds = {
            'PICNIC_BASKET1': 3.5,  # 更保守，防止亏损
            'PICNIC_BASKET2': 2.0   # 更激进，提高盈利
        }
        
        # 最小标准差阈值，防止初期过度交易
        self.min_std_threshold = 10.0

    # ——————工具函数——————

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

        elif amount < 0:
            for price, buy_amount in sorted(order_depth.buy_orders.items()):
                max_amount = min(buy_amount, position + self.position_limits[symbol], -amount)  # amount已经是负数，卖出
                if max_amount > 0:
                    # 卖出
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
    
    def calculate_basket_value(self, basket: str) -> float:
        """计算篮子理论价值 - 组件价格总和"""
        if not self.component_prices:
            #logger.print(f"No component prices available for {basket}")
            return 0
            
        components = self.basket_composition[basket]
        basket_value = sum(qty * self.component_prices.get(product, 0) for product, qty in components.items())
        
        # 添加额外的固定价值
        if basket == 'PICNIC_BASKET1':
            basket_value += 30
        elif basket == 'PICNIC_BASKET2':
            basket_value += 103
            
        #logger.print(f"Calculated {basket} theoretical value: {basket_value}")
        return basket_value
    
    def calculate_std_dev(self, basket: str) -> float:
        """计算给定篮子价差的标准差"""
        if len(self.price_diff_history[basket]) < 2:
            return 0
        return np.std(list(self.price_diff_history[basket]))
    # ———————下单模块——————

    #basket统一订单生成函数
    def generate_basket_orders(self, state: TradingState) -> List[Order]:
            """生成订单逻辑 - 实现基础篮子套利策略"""
            orders = []
            #logger.print(f"BasketStrategy.generate_orders - Starting for {self.symbol}")
            
            # 1. 更新组件价格
            for component in ['CROISSANTS', 'JAMS', 'DJEMBES']:
                if component in state.order_depths:
                    #logger.print(f"Processing component {component}")
                    component_price = self.calculate_fair_value(state.order_depths[component])
                    if component_price > 0:
                        self.component_prices[component] = component_price
                        #logger.print(f"Updated {component} price: {component_price}")
                    else:
                        logger.print(f"Failed to get valid price for {component}")
            
            #logger.print(f"Component prices: {self.component_prices}")
            
            # 2. 获取篮子价格和计算理论价值
            basket_prices = {}
            basket_values = {}
            price_diffs = {}
            
            for basket in ['PICNIC_BASKET1', 'PICNIC_BASKET2']:
                '''
                # 跳过非当前篮子（确保完全分离交易）
                if basket != self.symbol and self.symbol.startswith('PICNIC_BASKET'):
                    logger.print(f"Skipping {basket} as we're only trading {self.symbol}")
                    continue
                '''
                    
                # 计算篮子理论价值
                basket_value = self.calculate_basket_value(basket)
                if basket_value > 0:
                    basket_values[basket] = basket_value
                    #logger.print(f"{basket} theoretical value: {basket_value}")
                    
                    # 获取篮子市场价格
                    if basket in state.order_depths:
                        #logger.print(f"Processing basket {basket}")
                        basket_price = self.calculate_fair_value(state.order_depths[basket])
                        if basket_price > 0:
                            basket_prices[basket] = basket_price
                            #logger.print(f"{basket} market price: {basket_price}")
                            
                            # 计算价差并记录
                            price_diff = basket_price - basket_value
                            price_diffs[basket] = price_diff
                            #logger.print(f"{basket} price diff: {price_diff}")
                            
                            # 更新价差历史
                            self.price_diff_history[basket].append(price_diff)
                            
                            # 更新标准差 (现在是价差的标准差)
                            self.std_devs[basket] = max(self.calculate_std_dev(basket), self.min_std_threshold)
                            #logger.print(f"{basket} price diff std dev: {self.std_devs[basket]}, history size: {len(self.price_diff_history[basket])}")
                        else:
                            logger.print(f"Failed to get valid price for {basket}")

            
            # 3. 确定交易方向和数量
            for basket in ['PICNIC_BASKET1', 'PICNIC_BASKET2']:
                if basket in price_diffs and self.std_devs[basket] > 0:
                    current_position = state.position.get(basket, 0)
                    position_limit = self.position_limits[basket]
                    std_threshold = self.std_thresholds[basket]
                    
                    # 计算可用交易额度
                    available_buy = max(0, position_limit - current_position)
                    available_sell = max(0, position_limit + current_position)
                    
                    # 计算交易信号 - 使用篮子特定的标准差阈值
                    buy_signal = price_diffs[basket] < -std_threshold * self.std_devs[basket]  # 篮子低估，买入信号
                    sell_signal = price_diffs[basket] > std_threshold * self.std_devs[basket]  # 篮子高估，卖出信号
                    # 执行买入
                    if buy_signal and available_buy > 0 and basket in state.order_depths:
                        # 找出最佳卖价
                        sell_orders = sorted(state.order_depths[basket].sell_orders.items())
                        basket_orders = []
                        
                        # 遍历所有卖单，从最低价开始吃单
                        remaining_buy = available_buy
                        for price, volume in sell_orders:
                            # 卖单的volume是负数
                            buyable = min(remaining_buy, -volume)
                            if buyable > 0:
                                basket_orders.append(Order(basket, price, buyable))
                                #logger.print(f"BUY {basket}: {buyable} @ {price}")
                                remaining_buy -= buyable
                                if remaining_buy <= 0:
                                    break
                        
                        orders.extend(basket_orders)
                    
                    # 执行卖出
                    elif sell_signal and available_sell > 0 and basket in state.order_depths:
                        # 找出最佳买价
                        buy_orders = sorted(state.order_depths[basket].buy_orders.items(), reverse=True)
                        basket_orders = []
                        
                        # 遍历所有买单，从最高价开始吃单
                        remaining_sell = available_sell
                        for price, volume in buy_orders:
                            # 买单的volume是正数
                            sellable = min(remaining_sell, volume)
                            if sellable > 0:
                                basket_orders.append(Order(basket, price, -sellable))
                                #logger.print(f"SELL {basket}: {sellable} @ {price}")
                                remaining_sell -= sellable
                                if remaining_sell <= 0:
                                    break
                        
                        orders.extend(basket_orders)
            
            return orders
    def generate_orders_basket1(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        basket_orders = self.generate_basket_orders(state)
        orders = [basket_order for basket_order in basket_orders if basket_order.symbol == 'PICNIC_BASKET1']
        return orders

    def generate_orders_basket2(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        basket_orders = self.generate_basket_orders(state)
        orders = [basket_order for basket_order in basket_orders if basket_order.symbol != 'PICNIC_BASKET1']    
        return orders

    def generate_orders_croissant(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        # 处理Basket1的信号
        if len(self.delta1_history) >= self.std_window:
            delta1 = self.delta1_history[-1]
            if abs(delta1) > self.std_multiplier * np.std(self.delta1_history):
                # 根据篮子1的系数计算交易量（6:1）
                basket_trade_direction = 1 if delta1 > 0 else -1  # 篮子方向与组件相反
                component_amount = -6 * basket_trade_direction * self.get_max_possible_trade('PICNIC_BASKET1', state,
                                                                                              basket_trade_direction)
                orders.extend(self.quick_trade(symbol, state, component_amount)[0])

        # 处理Basket2的信号（类似逻辑）
        if len(self.delta2_history) >= self.std_window:
            delta2 = self.delta2_history[-1]
            if abs(delta2) > self.std_multiplier * np.std(self.delta2_history):
                basket_trade_direction = 1 if delta2 > 0 else -1
                component_amount = -4 * basket_trade_direction * self.get_max_possible_trade('PICNIC_BASKET2', state,
                                                                                              basket_trade_direction)
                orders.extend(self.quick_trade(symbol, state, component_amount)[0])
        return orders

    def generate_orders_jams(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        # Basket1相关
        if len(self.delta1_history) >= self.std_window:
            delta1 = self.delta1_history[-1]
            if abs(delta1) > self.std_multiplier * np.std(self.delta1_history):
                basket_trade_direction = 1 if delta1 > 0 else -1
                component_amount = -3 * basket_trade_direction * self.get_max_possible_trade('PICNIC_BASKET1', state,
                                                                                              basket_trade_direction)
                orders.extend(self.quick_trade(symbol, state, component_amount)[0])

        # Basket2相关
        if len(self.delta2_history) >= self.std_window:
            delta2 = self.delta2_history[-1]
            if abs(delta2) > self.std_multiplier * np.std(self.delta2_history):
                basket_trade_direction = 1 if delta2 > 0 else -1
                component_amount = -2 * basket_trade_direction * self.get_max_possible_trade('PICNIC_BASKET2', state,
                                                                                              basket_trade_direction)
                orders.extend(self.quick_trade(symbol, state, component_amount)[0])
        return orders

    def generate_orders_djembes(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        if len(self.delta1_history) >= self.std_window:
            delta1 = self.delta1_history[-1]
            if abs(delta1) > self.std_multiplier * np.std(self.delta1_history):
                basket_trade_direction = 1 if delta1 > 0 else -1
                component_amount = -1 * basket_trade_direction * self.get_max_possible_trade('PICNIC_BASKET1', state, basket_trade_direction)
                orders.extend(self.quick_trade(symbol, state, component_amount)[0])
        return orders

    def get_max_possible_trade(self, symbol: str, state: TradingState, direction: int) -> int:
        """计算最大可交易量（考虑仓位限制和订单簿深度）"""
        position = state.position.get(symbol, 0)
        available = self.position_limits[symbol] - abs(position)
        return min(available, 5) * direction  # 示例：每次最多交易5单位（可根据需要调整）

    def generate_orders(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        # 计算并记录当前delta
        self.delta1_history.append(self.get_price_delta_basket1(state))
        self.delta2_history.append(self.get_price_delta_basket2(state))

        # 生成订单（移除所有原有套利逻辑参数）
        orders = {}
        strategy_map = {
            'PICNIC_BASKET1': self.generate_orders_basket1,
            'PICNIC_BASKET2': self.generate_orders_basket2,
            'CROISSANTS': self.generate_orders_croissant,
            'JAMS': self.generate_orders_jams,
            'DJEMBES': self.generate_orders_djembes,
        }

        for symbol in self.symbols:
            if symbol in state.order_depths:
                orders[symbol] = strategy_map[symbol](symbol, state)

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

        if len(self.fair_value_history[symbol]) > self.std_window:
            self.fair_value_history[symbol] = self.fair_value_history[symbol][-self.std_window:]

        return

    def save_state(self, state):
        # 对每个产品维护历史数据
        for symbol in self.symbols:
            if symbol in state.order_depths:
                self.save_history(symbol, state)

        return {
            'price_diff_history': {
                basket: list(history) for basket, history in self.price_diff_history.items()
            },
            'std_devs': self.std_devs,
            'component_prices': self.component_prices
        }

    def load_state(self, state):
        if hasattr(state, 'traderData') and state.traderData:
            try:
                trader_data = json.loads(state.traderData)
                basket_data = trader_data.get(self.symbol, {})
                
                # 加载价差历史
                if 'price_diff_history' in basket_data:
                    for basket, history in basket_data['price_diff_history'].items():
                        self.price_diff_history[basket] = deque(history, maxlen=20)
                
                # 加载标准差
                if 'std_devs' in basket_data:
                    self.std_devs = basket_data['std_devs']
                
                # 加载组件价格
                if 'component_prices' in basket_data:
                    self.component_prices = basket_data['component_prices']
            except Exception as e:
                logger.print(f"Error loading state: {str(e)}")

class VolcanicRockStrategy(Strategy):
    def __init__(self, symbols: List[str], position_limits: dict, threshold: float = 10, time_window: int = 100):
        # 使用第一个symbol作为虚拟主产品
        super().__init__(symbols[0], position_limits[symbols[0]])
        self.symbols = symbols
        self.position_limits = position_limits
        self.time_window = time_window
        self.history = {}
        self.threshold = threshold
        self.T = 5/365

    #工具函数

    def calculate_mid_price(self, order_depth):
        """使用买卖加权的price计算mid_price"""

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

        mid_price = (buy_avg + sell_avg) / 2
        # 返回中间价
        return mid_price


    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    

    def bs_call_price(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes formula for call option price using numpy."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.norm_cdf(d1) - K * np.exp(-r * T) * self.norm_cdf(d2)
    
    def predict_iv_weighted(self, voucher_prices, underlying_prices, strike, T_list, decay=0.1):
        n = len(voucher_prices)
        if n < 5:
            return 0.2
        X, y, weights = [], [], []

        for i in range(n):
            S = underlying_prices[i]
            T = T_list[i]
            price = voucher_prices[i]
            if S <= 0 or T <= 0:
                continue
            X.append([S, T])
            y.append(price)
            # 越靠近现在的样本，index 越大，权重越大
            weights.append(math.exp(-decay * (n - 1 - i)))

        if len(X) < 5:
            return 0.2

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)

        # 加权线性拟合
        W = np.diag(weights)
        XTW = X.T @ W
        coeffs = np.linalg.inv(XTW @ X) @ XTW @ y
        est_vol = abs(coeffs[0]) / 10 
        return max(0.05, min(2.0, est_vol))

    def compute_call_delta(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        return self.norm_cdf(d1)
    

    def calculate_fair_value(self, symbol: str):
        """利用bs模型定价"""
        #从symbol中取出strike，例如VOLCANIC_ROCK_VOUCHER_9500取出9500
        strike = int(symbol.split("_")[-1])
        data = self.history[symbol]
        voucher_prices = data["mid_price_history"]
        underlying_prices = self.history["VOLCANIC_ROCK"]['mid_price_history']
        T_list = [self.T] * len(voucher_prices)
        current_T = T_list[-1]

        #拟合隐含波动率
        iv = self.predict_iv_weighted(voucher_prices, underlying_prices, strike, T_list)

        current_S = self.history["VOLCANIC_ROCK"]['mid_price_history'][-1]
        #BS 模型定价
        fair_price = self.bs_call_price(current_S, strike, current_T, r=0.0, sigma=iv)

        return fair_price
    
    def get_implied_volatility(self, symbol: str) -> float:
        prices = self.history[symbol]["mid_price_history"]
        if len(prices) < 2:
            return 0.2  # fallback

        returns = np.diff(np.log(prices))
        weights = np.linspace(0.1, 1.0, num=len(returns))
        weights /= weights.sum()

        weighted_var = np.sum(weights * (returns ** 2))
        sigma_tick = np.sqrt(weighted_var)

        # 年化：假设一天 10000 tick，一年 250 天
        sigma_annual = sigma_tick * np.sqrt(250 * 10000)
        return float(np.clip(sigma_annual, 0.05, 2.0))
    
    #下单函数

    def calculate_current_portfolio_delta(self, state: TradingState) -> float:
        total_delta = 0.0
        S = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK"])
        if S is None:
            return 0.0

        for symbol in self.symbols:
            if "VOUCHER" not in symbol:
                continue

            position = state.position.get(symbol, 0)
            if position == 0:
                continue

            K = int(symbol.split("_")[-1])
            sigma = self.get_implied_volatility(symbol)
            if sigma is None:
                continue

            delta = self.compute_call_delta(S, K, self.T, sigma)
            total_delta += delta * position

        return total_delta
    
    def generate_orders_fair_value_arbitrage(self, state: TradingState) -> Tuple[Dict[str, List[Order]], float]:
        orders = {}
        total_delta_exposure = 0.0  # 用局部变量，不要用 self 记录，避免多次调用累加错

        for symbol in self.symbols:
            if symbol == "VOLCANIC_ROCK":
                continue  # 跳过标的

            order_depth = state.order_depths[symbol]
            mid_price = self.calculate_mid_price(order_depth)
            fair_price = self.calculate_fair_value(symbol)
            position = state.position.get(symbol, 0)
            limit = self.position_limits[symbol]

            delta_p = fair_price - mid_price
            acceptable_threshold = self.threshold
            orders_for_symbol = []

            # 获取 strike 和基础价格
            K = int(symbol.split("_")[-1])
            S = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK"])
            sigma = self.get_implied_volatility(symbol)
            T = self.T

            # None check
            if mid_price is None or fair_price is None or S is None or sigma is None:
                continue

            option_delta = self.compute_call_delta(S, K, T, sigma)


            logger.print(f"current symbol {symbol}, mid_price {mid_price}, fair_price {fair_price}")
            logger.print(f"delta_p {delta_p}, option_delta {option_delta}")
            if delta_p > self.threshold and position < limit:
                volume = min(limit - position, 1)
                buy_price = int(mid_price + 1)
                orders_for_symbol.append(Order(symbol, buy_price, volume))
                total_delta_exposure += option_delta * volume

            elif delta_p < -self.threshold and position > -limit:
                volume = min(position + limit, 1)
                sell_price = int(mid_price - 1)
                orders_for_symbol.append(Order(symbol, sell_price, -volume))
                total_delta_exposure -= option_delta * volume

            if orders_for_symbol:
                orders[symbol] = orders_for_symbol

        return orders, total_delta_exposure
    
    def generate_orders_delta_hedge(self, state: TradingState, total_delta_exposure: float) -> List[Order]:
        orders = []
        symbol = "VOLCANIC_ROCK"
        position = state.position.get(symbol, 0)
        limit = self.position_limits[symbol]

        mid_price = self.calculate_mid_price(state.order_depths[symbol])
        if mid_price is None:
            return []

        target = -int(round(total_delta_exposure))  # delta hedge 理论目标
        diff = target - position

        if diff > 0:
            # 需要买入
            volume = min(diff, limit - position)
            if volume > 0:
                orders.append(Order(symbol, int(mid_price - 1), volume))
        elif diff < 0:
            # 需要卖出
            volume = min(-diff, position + limit)
            if volume > 0:
                orders.append(Order(symbol, int(mid_price + 1), -volume))

        return orders
    
    def generate_orders(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        #先检查当前仓位的hedge情况
        current_delta = self.calculate_current_portfolio_delta(state)

        arbitrage_orders, total_delta_exposure = self.generate_orders_fair_value_arbitrage(state)
        hedge_orders = self.generate_orders_delta_hedge(state, total_delta_exposure + current_delta)

        orders.update(arbitrage_orders)
        if hedge_orders:
            orders["VOLCANIC_ROCK"] = hedge_orders

        return orders
    

    def load_state(self, state):
        
        #储存每个产品的mid_price
        rock_mid_price = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK"])
        voucher_9500_mid_price = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK_VOUCHER_9500"])
        voucher_9750_mid_price = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK_VOUCHER_9750"])
        voucher_10000_mid_price = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK_VOUCHER_10000"])
        voucher_10250_mid_price = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK_VOUCHER_10250"])
        voucher_10500_mid_price = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK_VOUCHER_10500"])
        
        mid_prices = {
            "VOLCANIC_ROCK": rock_mid_price,
            "VOLCANIC_ROCK_VOUCHER_9500": voucher_9500_mid_price,
            "VOLCANIC_ROCK_VOUCHER_9750": voucher_9750_mid_price,
            "VOLCANIC_ROCK_VOUCHER_10000": voucher_10000_mid_price,
            "VOLCANIC_ROCK_VOUCHER_10250": voucher_10250_mid_price,
            "VOLCANIC_ROCK_VOUCHER_10500": voucher_10500_mid_price,
        }

        
        for symbol in self.symbols:
            if symbol not in self.history:
                self.history[symbol] = {}
                self.history[symbol]["mid_price_history"] = deque(maxlen=self.time_window)
            
            self.history[symbol]["mid_price_history"].append(mid_prices[symbol])
        
        for symbol in self.symbols:
            if len(self.history[symbol]["mid_price_history"]) > self.time_window:
                self.history[symbol]["mid_price_history"] = self.history[symbol]["mid_price_history"][-self.time_window:]


    def save_state(self, state):
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
            "VOLCANIC_ROCK_GROUP":{
                "strategy_cls": VolcanicRockStrategy,
                "symbols": ["VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"],
                "position_limits": {
                    "VOLCANIC_ROCK": 400,
                    "VOLCANIC_ROCK_VOUCHER_9500": 200,
                    "VOLCANIC_ROCK_VOUCHER_9750": 200,
                    "VOLCANIC_ROCK_VOUCHER_10000": 200,
                    "VOLCANIC_ROCK_VOUCHER_10250": 200,
                    "VOLCANIC_ROCK_VOUCHER_10500": 200,
                },
                "threshold": 10,
                "time_window": 100,
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
