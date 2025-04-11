from abc import ABC, abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict, List, Optional, Tuple, Deque
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

        fair_value = fair_value -0.03 * current_position

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

    def __init__(self, symbol: str, position_limit: int, base_offset: int, level2spread: int):
        super().__init__(symbol, position_limit)
        # 策略参数
        self.base_offset = base_offset
        self.level2spread = level2spread

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
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

    def generate_orders(self, state: TradingState) -> List[Order]:
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        orders = []
        offset = self.base_offset
        take_position1 = 0
        take_position2 = 0
        level2spread = self.level2spread
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
            elif ask == FIXED_MID and available_sell< 30:
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
        desired_buy = available_buy - take_position1
        desired_sell = available_sell - take_position2  # 固定吃单额度

        # 买盘挂单（正数表示买入）
        if desired_buy > 0 :
            orders.append(Order(self.symbol, desired_bid, desired_buy))

        # 卖盘挂单（负数表示卖出）
        if desired_sell > 0 :
            orders.append(Order(self.symbol, desired_ask, -desired_sell))

        return orders


    def save_state(self, state) -> dict:
        return {}

    def load_state(self, state):
        pass


class SquidInkStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int, ma_window: int = 200,
                max_deviation: int = 200, vol_threshold: float = 10, band_width: float = 25,
                trend_window: int = 100, take_spread: float = 10, break_step: float = 10, fallback_threshold: float = 0.1):
        super().__init__(symbol, position_limit)

        self.timestamp = 0

        #策略参数
        self.ma_window = ma_window
        self.max_deviation = max_deviation
        self.vol_threshold = vol_threshold
        self.band_width = band_width
        self.trend_window = trend_window
        self.take_spread = take_spread
        self.break_step = break_step
        self.fallback_threshold = fallback_threshold

        #策略历史数据
        self.fair_value_history = deque(maxlen=ma_window)
        self.fair_value_ma200_history = deque(maxlen=ma_window)
        self.current_mode = "market_making"
        self.breakout_price: Optional[float] = None
        self.prepared_reverse = False
        self.max_breakout_distance = 0
        
        self.ma_short = 0

        self.breakout_times = 0 

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
        orders = []
        order_depth = state.order_depths[self.symbol]
        buy_orders = [(p, v) for p, v in order_depth.buy_orders.items() if v > 0]
        sell_orders = [(p, v) for p, v in order_depth.sell_orders.items() if v > 0]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
            
        best_bid_amount = order_depth.buy_orders[best_bid]
        best_ask_amount = order_depth.sell_orders[best_ask]


        position = state.position.get(self.symbol, 0)
        fair_value = self.calculate_fair_value(order_depth)

    
        vol_10 = np.std(list(self.fair_value_history)[-10:])

        logger.print("Current mode: ", self.current_mode)
        # Strategy 1: Market making
        if self.current_mode == "market_making":
            if len(self.fair_value_ma200_history) < 200 or abs(fair_value - self.fair_value_ma200_history[-1]) <= self.band_width:
                orders = []
                # 获取当前市场数据
                order_depth = state.order_depths[self.symbol]
                current_position = state.position.get(self.symbol, 0)
                max_position = self.position_limit
                logger.print(f"fair_value: {fair_value}, current_position: {current_position}, max_position: {max_position}")

                if len(self.fair_value_history) >= self.trend_window:
                    window_data = list(self.fair_value_history)[-self.trend_window:]
                    self.ma_short = np.mean(window_data)
                else:
                    fair_value = self.calculate_fair_value(order_depth)
                    self.ma_short = fair_value
                logger.print(f"ma_short: {self.ma_short}")

                available_buy = max(0, max_position - current_position)
                available_sell = max(0, max_position + current_position)
                logger.print(f"available_buy: {available_buy}, available_sell: {available_sell}")

                # 处理卖单（asks）的限价单
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                    if ask_price < (self.ma_short - self.take_spread):
                        quantity = min(-ask_volume, available_buy)
                        if quantity > 0:
                            orders.append(Order(self.symbol, ask_price, quantity))
                            available_buy -= quantity
                            logger.print(f"buy {quantity} at {ask_price}")

                # 处理买单（bids）的限价单
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > (self.ma_short + self.take_spread):
                        quantity = min(bid_volume, available_sell)
                        if quantity > 0:
                            orders.append(Order(self.symbol, bid_price, -quantity))
                            available_sell -= quantity
                            logger.print(f"sell {quantity} at {bid_price}")

                # 挂出被动做市单
                fair_value = self.ma_short

                # 计算挂单价格
                buy_price = math.floor(fair_value - self.take_spread)
                sell_price = math.ceil(fair_value + self.take_spread)
                logger.print(f"take_spread: {self.take_spread}, buy_price: {buy_price}, sell_price: {sell_price}")

                # 确保不超过仓位限制
                if available_buy > 0:
                    orders.append(Order(self.symbol, buy_price, available_buy))
                    logger.print(f"take_spread, buy {available_buy} at {buy_price}")

                if available_sell > 0:
                    orders.append(Order(self.symbol, sell_price, -available_sell))
                    logger.print(f"take_spread, sell {available_sell} at {sell_price}")

                return orders

            elif all(x != 0 for x in self.fair_value_ma200_history):
                self.breakout_price = fair_value
                self.breakout_times += 1
                #反向吃满
                self.direction = 1 if fair_value - self.fair_value_ma200_history[-1] else -1 #记录突破方向
                
                logger.print(f"Break! Breakout price: {self.breakout_price} Break direction {self.direction}")

                #反向吃满
                if self.direction == 1:
                    #突破是向上的，先做多
                    for price, amount in sorted(order_depth.sell_orders.items()):
                        max_amount = min(-amount, self.position_limit - position)
                        if max_amount > 0:
                            orders.append(Order(self.symbol, price, max_amount))
                            logger.print(f"Up break, buy {max_amount} at {price}")


                if self.direction == -1:
                    #突破是向下的，先做空
                    for price, amount in sorted(order_depth.buy_orders.items()):
                        max_amount = min(amount, position - self.position_limit)
                        if max_amount > 0:
                            orders.append(Order(self.symbol, price, -max_amount))
                            logger.print(f"Down break, sell {max_amount} at {price}")

                self.current_mode = "trend_following"

        # Strategy 2: Breakout
        elif self.current_mode == "trend_following" and self.breakout_price is not None:
            distance = fair_value - self.breakout_price
            #记录最大突破距离：
            if abs(distance) > self.max_breakout_distance + self.break_step:
                self.max_breakout_distance = abs(distance)
            self.direction = 1 if distance > 0 else -1 #往上突破为1 往下突破为0
            position = state.position.get(self.symbol, 0)

            #判断价格是否回归
            logger.print(f"Current distance: {(fair_value - self.breakout_price) * self.direction}, distance_threshold: {vol_10 * self.fallback_threshold}")
            # 回归就清仓
            if (fair_value - self.breakout_price) * self.direction < vol_10 * self.fallback_threshold:
                logger.print(f"Fall back! {fair_value}")
                if position != 0:
                    logger.print(f"Close position {position}")
                    if self.direction == 1:
                        #突破是向上的，平空
                        max_amount = min(best_bid_amount, -position)
                        orders.append(Order(self.symbol, best_bid + 1, max_amount))
        
                    if self.direction == -1:
                        #突破是向下的，平多
                        max_amount = min(best_ask_amount, position)
                        orders.append(Order(self.symbol, best_ask - 1, -max_amount))

                if position == 0:
                    logger.print(f"Back to market making mode")
                    #重置突破参数
                    self.breakout_price = None
                    self.prepared_reverse = False
                    self.direction = 0
                    self.max_breakout_distance = 0
                    self.current_mode = "market_making"

            #如果没有回归，吃回调
            else:
                #先检查仓位有没有反向吃满，如果没有则先吃满。注意只能做一次，不然会反复反向吃满
                if position * self.direction < self.position_limit and not self.prepared_reverse:
                    logger.print(f"Preparing reverse, current position {position}, direction {self.direction}")
                    logger.print(f"{self.position_limit - position} to fill")
                    if self.direction == 1:
                        #突破是向上的，先做多
                        for price, amount in sorted(order_depth.sell_orders.items()):
                            max_amount = min(-amount, self.position_limit - position)
                            if max_amount > 0:
                                orders.append(Order(self.symbol, price, max_amount))


                    if self.direction == -1:
                        #突破是向下的，先做空
                        for price, amount in sorted(order_depth.buy_orders.items()):
                            max_amount = min(amount, position - self.position_limit)
                            if max_amount > 0:
                                orders.append(Order(self.symbol, price, -max_amount))
                
                else:
                    logger.print(f"Starting Reverse")
                    self.prepared_reverse = True #反向吃满了就设置为True
                    #只有吃满了仓位才开始反转
                    target_position = -self.direction * self.position_limit
                    delta_position = target_position - position #还要做多少仓位才到顶
                    
                    current_position = state.position.get(self.symbol, 0)
                    if delta_position != 0 and abs(distance) >= self.max_breakout_distance: #只有当价格突破新高(10)的时候才下单
                        res_position = self.position_limit - position if self.direction == 1 else position + self.position_limit
                        amount = min(int(abs(distance) * self.direction * delta_position / self.max_deviation), res_position)
                        #注意amount已经包括了direction
                        if self.direction == 1:
                            orders.append(Order(self.symbol, best_ask - 1, amount))
                        if self.direction == -1:
                            orders.append(Order(self.symbol, best_bid + 1, amount))
        return orders

    def save_state(self, state):
        return {}

    def load_state(self, state):
        fair_value = self.calculate_fair_value(state.order_depths[self.symbol])
        self.fair_value_history.append(fair_value)
        if len(self.fair_value_history) > self.ma_window:
            self.fair_value_history.popleft()

        fair_value_ma200 = np.mean(list(self.fair_value_history)[-200:]) if len(self.fair_value_history) >= 200 else 0
        self.fair_value_ma200_history.append(fair_value_ma200)
        if len(self.fair_value_ma200_history) > self.ma_window:
            self.fair_value_ma200_history.popleft()
        
        pass




# 组合策略
class BasketStrategy(Strategy):
    def __init__(self, main_symbol: str, symbols: List[str], position_limits: dict):
        super().__init__(main_symbol, position_limits[main_symbol])
        self.symbols = symbols
        self.position_limits = position_limits

    # 处理CROISSANT
    def generate_orders_croissant(self, state: TradingState) -> List[Order]:
        orders = []

    #其他产品类似
        
    def generate_orders(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        orders = {}
        strategy_map = {
        'CROISSANT': self.generate_orders_croissant,
        'RAINFOREST_RESIN': self.generate_orders_resin,
        'KELP': self.generate_orders_kelp,
        # 添加其他产品映射...
    }
        # 遍历处理所有相关产品
        for symbol in self.symbols:
            if symbol in state.order_depths:
                # 具体订单生成逻辑
                order_depth = state.order_depths[symbol]
                position = state.position.get(symbol, 0)
                # 生成该symbol的订单...
                handler = strategy_map.get(symbol, self.generate_default_orders)
                orders[symbol] = handler(state, symbol)

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], dict]:
        orders = self.generate_orders(state)
        strategy_state = self.save_state(state)
        return orders, strategy_state
    
    def save_state(self, state):
        return super().save_state(state)
    
    def load_state(self, state):
        return super().load_state(state)
    
    


class Config:
    def __init__(self):
        self.PRODUCT_CONFIG = {
        "KELP": {
            "strategy_cls": KelpStrategy,
            "position_limit": 50,
            "alpha": 0,
            "beta": 0
        },
        "RAINFOREST_RESIN": {
            "strategy_cls": RainforestResinStrategy,
            "position_limit": 50,  # 最大持仓
            "base_offset": 3,  # 基础报价偏移
            "level2spread": 8,  # spread超过这个值就用另一个offset
        },
        "SQUID_INK": {
            "strategy_cls": SquidInkStrategy,
            "position_limit": 50,          # 最大持仓量
            "ma_window": 200,          # 计算均价的时长
            "max_deviation": 200,       # 偏离标准距离（最大距离）      
            "band_width": 30,          # 波动率计算的宽度
            "trend_window": 100,       # 趋势判断的时长
            "take_spread": 10,          #market making mode take width
            "break_step": 15,           #price range to next reverse order
            "fallback_threshold": 0,   #price range to fall back * vol_10
        },
        "BASKET": {
            "strategy_cls": BasketStrategy,
            "sub_products": ["BASKET1", "BASKET2", "CROISSANT", "JAM", "DJEMBE"],
            "basket1_position_limit": 60,
            "basket2_position_limit": 100,
            "cross_position_limit": 50,
            "croissant_position_limit": 50, 
            "jam_position_limit": 350,
            "djembe_position_limit": 60,
        },
    }
        

class Trader:
    def _init_strategies(self):
        for product, config in self.PRODUCT_CONFIG.items():
            if "sub_products" in config:  # 处理组合产品
                strategy = config["strategy_cls"](
                    main_symbol=product,
                    symbols=[product] + config["sub_products"],
                    **config
                )
                # 注册主产品和所有子产品
                for symbol in [product] + config["sub_products"]:
                    self.strategies[symbol] = strategy
            else:
                # 常规产品初始化
                cls = config["strategy_cls"]
                args = {k:v for k,v in config.items() if k != "strategy_cls"}
                self.strategies[product] = cls(symbol=product, **args)

class Trader:
    def __init__(self, product_config=None):
        # 使用默认 config，或外部传入 config
        self.PRODUCT_CONFIG = product_config if product_config is not None else Config().PRODUCT_CONFIG
        self.strategies = {}
        self._init_strategies()

    def _init_strategies(self):
        config = Config()
        for product, config in self.PRODUCT_CONFIG.items():
            cls = config["strategy_cls"]
            args = {k: v for k, v in config.items() if k != "strategy_cls"}
            self.strategies[product] = cls(symbol=product, **args)

    def run(self, state: TradingState):
        conversions = 0
        # 加载历史状态
        trader_data = json.loads(state.traderData) if state.traderData else {}

        orders = {}
        new_trader_data = {}

        for product, strategy in self.strategies.items():
            if product in trader_data:
                strategy.load_state(state)
            if product in state.order_depths:
                product_orders, strategy_state = strategy.run(state)
                orders[product] = product_orders
                
                new_trader_data[product] = strategy_state

        trader_data.update(new_trader_data)
        trader_data = json.dumps(trader_data)
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data
