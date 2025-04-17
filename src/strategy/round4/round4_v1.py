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


class KelpStrategy(Strategy):
    """海带做市策略"""

    def __init__(self, symbol: str, position_limit: int, alpha: float, beta: float):
        super().__init__(symbol, position_limit)
        # 添加海带策略特有参数
        self.alpha = alpha  # adjusted fair price清仓系数
        self.beta = beta  # adjusted fair price订单簿不平衡度系数
        self.trader_data = {}
        self.position_history = []

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        """使用买卖加权的price计算fair_value"""

        def weighted_avg(prices_vols, n=3):
            """使用NumPy计算加权平均价格"""
            # 按价格排序（买单降序，卖单升序）
            is_buy = isinstance(prices_vols, dict) and len(prices_vols) > 0 and next(iter(prices_vols.values())) > 0
            sorted_orders = sorted(prices_vols.items(), key=lambda x: x[0], reverse=is_buy)[:n]

            if not sorted_orders:
                return 0

            # 转换为NumPy数组进行计算
            prices = np.array([price for price, _ in sorted_orders])
            volumes = np.array([abs(vol) for _, vol in sorted_orders])

            # 计算加权平均价
            if volumes.sum() > 0:
                return np.sum(prices * volumes) / volumes.sum()
            return 0

        # 计算买卖方加权均价
        buy_avg = weighted_avg(order_depth.buy_orders)
        sell_avg = weighted_avg(order_depth.sell_orders)
        return (buy_avg + sell_avg) / 2

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
        desired_buy = available_buy - take_position1
        desired_sell = available_sell - take_position2

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
    def __init__(self, symbol: str, position_limit: int, ma_window: int = 200):
        super().__init__(symbol, position_limit)

        # 策略参数
        self.ma_window = ma_window
        # 策略历史数据
        self.fair_value_history = deque(maxlen=ma_window)
        self.fair_value_ma200_history = deque(maxlen=ma_window)
        self.current_mode = "No_action"
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
        orders = []
        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        fair_value = self.calculate_fair_value(state.order_depths[self.symbol])
        # 保存fair_value
        self.fair_value_history.append(fair_value)
        if len(self.fair_value_history) > self.ma_window:
            self.fair_value_history.popleft()

        if len(self.fair_value_history) >= 200:
            ma = np.mean(list(self.fair_value_history)[-200:])
        else:
            ma = fair_value

        available_buy = 50 - current_position
        available_sell = 50 + current_position

        ## ================================= 正式策略 ================================= ##
        # needle mode 时
        if (self.current_mode == "action"):
            # 无论仓位是否满，只要价格回归优先平仓
            if fair_value >= ma + 10 and current_position > 0 and self.needle_direction == -1:
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price >= ma - 10:
                        quantity = min(bid_volume, current_position)
                        orders.append(Order(self.symbol, bid_price, -quantity))
                        current_position -= quantity
                        if current_position == 0: break
                return orders

            elif fair_value <= ma + 10 and current_position < 0 and self.needle_direction == 1:
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                    if ask_price <= ma - 10:
                        quantity = min(-ask_volume, -current_position)
                        orders.append(Order(self.symbol, ask_price, quantity))
                        current_position += quantity
                        if current_position == 0: break
                return orders

                # 重置状态
            if current_position == 0:
                self.current_mode = "No_action"
                self.needle_direction = 0
                return orders
            # 如果仓位还没满，且价格持续下跌
            elif abs(current_position) < self.position_limit:
                # 持续吃单直到满仓
                if self.needle_direction == -1:  # 下跌插针
                    for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                        if ask_price < ma - 30:
                            quantity = min(-ask_volume, available_buy)
                            if quantity > 0 and current_position + quantity < 45:  # 仓位控制
                                orders.append(Order(self.symbol, ask_price, quantity))
                                available_buy -= quantity
                        if ask_price < ma - 120:  # 极端情况，少量仓位拉低均价
                            quantity = min(-ask_volume, available_buy)
                            if quantity > 0:
                                orders.append(Order(self.symbol, ask_price, quantity))
                                available_buy -= quantity


                elif self.needle_direction == 1:  # 上涨插针
                    for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                        if bid_price > ma + 30:
                            quantity = min(bid_volume, available_sell)
                            if quantity > 0 and current_position + quantity > -45:  # 仓位控制
                                orders.append(Order(self.symbol, bid_price, -quantity))
                                available_sell -= quantity
                        if bid_price > ma + 120:  # 极端情况，少量仓位抬高均价
                            quantity = min(bid_volume, available_sell)
                            if quantity > 0:
                                orders.append(Order(self.symbol, bid_price, -quantity))
                                available_sell -= quantity
            return orders
        # 波幅超标检测
        else:
            # 下跌插针检测
            if best_ask < ma - 30:  # 已检测到波动
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                    if ask_price < ma - 30:  # 低点
                        quantity = min(-ask_volume, available_buy)
                        if quantity > 0:
                            orders.append(Order(self.symbol, ask_price, quantity))
                            available_buy -= quantity
                self.current_mode = "action"
                self.needle_direction = -1
                return orders

            # 上涨插针检测
            elif best_bid > ma + 30:  # 已检测到波动
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > ma + 30:  # 高点
                        quantity = min(bid_volume, available_sell)
                        if quantity > 0:
                            orders.append(Order(self.symbol, bid_price, -quantity))
                            available_sell -= quantity
                self.current_mode = "action"
                self.needle_direction = 1
                return orders
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
        self.position_limits = position_limits

        self.delta1_history = deque(maxlen=time_window)
        self.delta2_history = deque(maxlen=time_window)
        self.std_window = std_window
        self.delta1_threshold = delta1_threshold
        self.delta2_threshold = delta2_threshold
        self.time_window = time_window
        self.std_multiplier = 1.4

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

        # 篮子价格和理论价值历史，用于计算相关性
        self.basket_price_history = {
            'PICNIC_BASKET1': deque(maxlen=30),
            'PICNIC_BASKET2': deque(maxlen=30)
        }
        self.basket_value_history = {
            'PICNIC_BASKET1': deque(maxlen=30),
            'PICNIC_BASKET2': deque(maxlen=30)
        }

        # EWMA相关性历史和权重
        self.ewma_alpha = 0.2  # EWMA衰减系数，值越大表示越重视近期数据
        self.ewma_correlation = {
            'PICNIC_BASKET1': 0.8,  # 初始默认相关性
            'PICNIC_BASKET2': 0.8
        }

        # 相关性最低门槛，低于此值时不开新仓
        self.min_correlation_threshold = 0.65

        # 组件价格缓存
        self.component_prices = {}

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

    def get_price_delta(self, state: TradingState, basket: str) -> float:
        """
        计算篮子和其组分的价差：delta = basket_price - components_value

        Args:
            state: 当前交易状态
            basket: 篮子名称，如'PICNIC_BASKET1'或'PICNIC_BASKET2'

        Returns:
            价差delta，如果缺少必要数据则返回0
        """
        # 获取篮子组成
        if basket not in self.basket_composition:
            return 0.0

        components = self.basket_composition[basket]

        # 检查所有所需产品是否在order_depths中
        required_products = [basket] + list(components.keys())
        for product in required_products:
            if product not in state.order_depths:
                return 0.0  # 如果缺少任何必要产品，返回0

        # 计算篮子自身价格
        basket_fair_value = self.calculate_fair_value(state.order_depths[basket])

        # 计算组件总价值
        components_value = 0
        for component, qty in components.items():
            component_fair_value = self.calculate_fair_value(state.order_depths[component])
            if component_fair_value == 0:
                return 0.0  # 如果组件价格为0，返回0
            components_value += qty * component_fair_value

        # 加上固定价值
        if basket == 'PICNIC_BASKET1':
            components_value += 30
        elif basket == 'PICNIC_BASKET2':
            components_value += 103

        # 返回价差
        return basket_fair_value - components_value

    def calculate_basket_value(self, state: TradingState, basket: str) -> float:
        """计算篮子理论价值 - 组件价格总和"""
        if basket not in self.basket_composition:
            return 0.0

        components = self.basket_composition[basket]

        # 直接从当前市场价格计算
        basket_value = 0
        for component, qty in components.items():
            if component in state.order_depths:
                component_price = self.calculate_fair_value(state.order_depths[component])
                if component_price > 0:
                    basket_value += qty * component_price
                else:
                    # 如果组件价格无效，尝试使用缓存的价格
                    basket_value += qty * self.component_prices.get(component, 0)
            else:
                # 使用缓存价格
                basket_value += qty * self.component_prices.get(component, 0)

        # 添加额外的固定价值
        if basket == 'PICNIC_BASKET1':
            basket_value += 30
        elif basket == 'PICNIC_BASKET2':
            basket_value += 103

        return basket_value

    # ———————下单模块——————

    # basket统一订单生成函数
    def generate_basket_orders(self, state: TradingState) -> List[Order]:
        """生成订单逻辑 - 实现基础篮子套利策略"""
        orders = []

        # 1. 更新组件价格
        for component in ['CROISSANTS', 'JAMS', 'DJEMBES']:
            if component in state.order_depths:
                component_price = self.calculate_fair_value(state.order_depths[component])
                if component_price > 0:
                    self.component_prices[component] = component_price
                else:
                    logger.print(f"Failed to get valid price for {component}")

        # 2. 获取篮子价格和计算理论价值
        basket_prices = {}
        basket_values = {}
        price_diffs = {}

        for basket in ['PICNIC_BASKET1', 'PICNIC_BASKET2']:
            # 计算篮子理论价值
            basket_value = self.calculate_basket_value(state, basket)
            basket_values[basket] = basket_value
            basket_price = self.calculate_fair_value(state.order_depths[basket])
            basket_prices[basket] = basket_price

            # 记录价格和理论价值历史
            self.basket_price_history[basket].append(basket_price)
            self.basket_value_history[basket].append(basket_value)

            # 计算价差并记录
            price_diff = basket_price - basket_value
            price_diffs[basket] = price_diff
            self.price_diff_history[basket].append(price_diff)

            # 计算EWMA相关性
            ewma_corr = self.calculate_ewma_correlation(basket)

        current_diff = price_diffs[basket]
        # 添加长度检查，避免IndexError
        entry_diff = self.price_diff_history[basket][-2] if len(self.price_diff_history[basket]) >= 2 else current_diff
        stop_loss_threshold = entry_diff * 1.2  # 例如，价差翻倍则止损

        if (current_diff < 0 and current_diff < stop_loss_threshold) or (
                current_diff > 0 and current_diff > stop_loss_threshold):
            # 平仓逻辑：反向操作当前仓位
            position = state.position.get(basket, 0)
            if position != 0:
                target_amount = -position
                orders.extend(self.quick_trade(basket, state, target_amount)[0])
                return orders  # 优先处理止损

        # 3. 确定交易方向和数量
        for basket in ['PICNIC_BASKET1', 'PICNIC_BASKET2']:
            if basket in price_diffs:
                current_position = state.position.get(basket, 0)
                position_limit = self.position_limits[basket]

                # 获取当前EWMA相关性
                ewma_corr = self.ewma_correlation.get(basket, 0.8)

                # 计算可用交易额度
                available_buy = max(0, position_limit - current_position)
                available_sell = max(0, position_limit + current_position)

                # 根据相关性调整交易量
                volume_factor = min(1.0, max(0.3, ewma_corr))  # 将相关性映射到0.3-1.0范围的交易量因子
                max_buy_volume = int(available_buy * volume_factor)
                max_sell_volume = int(available_sell * volume_factor)

                buy_signal = False
                sell_signal = False
                remaining_buy = max_buy_volume
                remaining_sell = max_sell_volume

                # 计算交易信号 - 使用篮子特定的标准差阈值
                if basket == 'PICNIC_BASKET1':
                    buy_signal = price_diffs[basket] < -35  # 篮子低估，买入信号
                    sell_signal = price_diffs[basket] > 35  # 篮子高估，卖出信号

                if basket == 'PICNIC_BASKET2':
                    buy_signal = price_diffs[basket] < -40  # 篮子低估，买入信号
                    sell_signal = price_diffs[basket] > -2  # 篮子高估，卖出信号

                # 相关性过滤器: 相关性低于阈值时不开新仓
                if ewma_corr < self.min_correlation_threshold and current_position == 0:
                    continue

                # 执行买入
                if buy_signal and max_buy_volume > 0 and basket in state.order_depths:
                    # 找出最佳卖价
                    sell_orders = sorted(state.order_depths[basket].sell_orders.items())
                    basket_orders = []

                    for price, volume in sell_orders:
                        # 卖单的volume是负数
                        buyable = min(remaining_buy, -volume)
                        if buyable > 0:
                            basket_orders.append(Order(basket, price, buyable))
                            remaining_buy -= buyable
                            if remaining_buy <= 0:
                                break

                    orders.extend(basket_orders)

                # 执行卖出
                elif sell_signal and max_sell_volume > 0 and basket in state.order_depths:
                    # 找出最佳买价
                    buy_orders = sorted(state.order_depths[basket].buy_orders.items(), reverse=True)
                    basket_orders = []

                    # 遍历所有买单，从最高价开始吃单
                    for price, volume in buy_orders:
                        # 买单的volume是正数
                        sellable = min(remaining_sell, volume)
                        if sellable > 0:
                            basket_orders.append(Order(basket, price, -sellable))
                            remaining_sell -= sellable
                            if remaining_sell <= 0:
                                break

                    orders.extend(basket_orders)

        return orders

    def generate_orders_basket1(self, symbol: str, state: TradingState) -> List[Order]:
        basket_orders = self.generate_basket_orders(state)
        orders = [basket_order for basket_order in basket_orders if basket_order.symbol == 'PICNIC_BASKET1']
        return orders

    def generate_orders_basket2(self, symbol: str, state: TradingState) -> List[Order]:
        basket_orders = self.generate_basket_orders(state)
        orders = [basket_order for basket_order in basket_orders if basket_order.symbol != 'PICNIC_BASKET1']
        return orders

    def _get_max_possible_trade(self, symbol: str, state: TradingState, direction: int) -> int:
        """计算最大可交易量（考虑仓位限制和订单簿深度）"""
        position = state.position.get(symbol, 0)
        available = self.position_limits[symbol] - abs(position)
        return min(available, 15) * direction  # 示例：每次最多交易5单位（可根据需要调整）

    def _apply_correlation_filter(self, basket: str) -> Tuple[bool, float, float]:
        """
        根据相关性决定是否交易及如何调整交易参数
        返回：(是否交易, 数量调整系数, 调整后阈值系数)
        """
        ewma_corr = self.ewma_correlation.get(basket, 0.8)

        # 相关性过低时不交易
        if ewma_corr < self.min_correlation_threshold:
            return False, 0, 0

        # 计算交易量调整系数 (0.3-1.0)
        volume_factor = min(1.0, max(0.3, ewma_corr))

        # 计算阈值调整系数
        if ewma_corr > 0.9:
            threshold_factor = 0.6  # 相关性高，降低阈值更积极交易
        elif ewma_corr < 0.85:
            threshold_factor = 1.2  # 相关性低，提高阈值更保守交易
        else:
            threshold_factor = 1.0  # 保持默认阈值

        return True, volume_factor, threshold_factor

    def generate_orders_croissant(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        # 处理Basket1的信号
        if len(self.delta1_history) >= self.std_window:
            # 确保delta1_history不为空
            delta1 = self.delta1_history[-1] if self.delta1_history else 0
            std_dev1 = np.std(self.delta1_history)

            # 应用相关性过滤器
            should_trade, volume_factor, threshold_factor = self._apply_correlation_filter('PICNIC_BASKET1')

            if should_trade and abs(delta1) > self.std_multiplier * threshold_factor * std_dev1:
                # 根据篮子1的系数计算交易量（6:1），并应用相关性调整
                basket_trade_direction = 1 if delta1 > 0 else -1  # 篮子方向与组件相反
                base_amount = 6 * basket_trade_direction * self._get_max_possible_trade('PICNIC_BASKET1', state,
                                                                                        basket_trade_direction)
                component_amount = -int(base_amount * volume_factor)  # 应用量调整系数

                if component_amount != 0:
                    orders.extend(self.quick_trade(symbol, state, component_amount)[0])

        # 处理Basket2的信号（类似逻辑）
        if len(self.delta2_history) >= self.std_window:
            # 确保delta2_history不为空
            delta2 = self.delta2_history[-1] if self.delta2_history else 0
            std_dev2 = np.std(self.delta2_history)

            # 应用相关性过滤器
            should_trade, volume_factor, threshold_factor = self._apply_correlation_filter('PICNIC_BASKET2')

            if should_trade and abs(delta2) > self.std_multiplier * threshold_factor * std_dev2:
                basket_trade_direction = 1 if delta2 > 0 else -1
                base_amount = 4 * basket_trade_direction * self._get_max_possible_trade('PICNIC_BASKET2', state,
                                                                                        basket_trade_direction)
                component_amount = -int(base_amount * volume_factor)  # 应用量调整系数

                if component_amount != 0:
                    orders.extend(self.quick_trade(symbol, state, component_amount)[0])

        return orders

    def generate_orders_jams(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []
        # Basket1相关
        if len(self.delta1_history) >= self.std_window:
            # 确保delta1_history不为空
            delta1 = self.delta1_history[-1] if self.delta1_history else 0
            std_dev1 = np.std(self.delta1_history)

            # 应用相关性过滤器
            should_trade, volume_factor, threshold_factor = self._apply_correlation_filter('PICNIC_BASKET1')

            if should_trade and abs(delta1) > self.std_multiplier * threshold_factor * std_dev1:
                basket_trade_direction = 1 if delta1 > 0 else -1
                base_amount = 3 * basket_trade_direction * self._get_max_possible_trade('PICNIC_BASKET1', state,
                                                                                        basket_trade_direction)
                component_amount = -int(base_amount * volume_factor)  # 应用量调整系数

                if component_amount != 0:
                    orders.extend(self.quick_trade(symbol, state, component_amount)[0])

        # Basket2相关
        if len(self.delta2_history) >= self.std_window:
            # 确保delta2_history不为空
            delta2 = self.delta2_history[-1] if self.delta2_history else 0
            std_dev2 = np.std(self.delta2_history)

            # 应用相关性过滤器
            should_trade, volume_factor, threshold_factor = self._apply_correlation_filter('PICNIC_BASKET2')

            if should_trade and abs(delta2) > self.std_multiplier * threshold_factor * std_dev2:
                basket_trade_direction = 1 if delta2 > 0 else -1
                base_amount = 2 * basket_trade_direction * self._get_max_possible_trade('PICNIC_BASKET2', state,
                                                                                        basket_trade_direction)
                component_amount = -int(base_amount * volume_factor)  # 应用量调整系数

                if component_amount != 0:
                    orders.extend(self.quick_trade(symbol, state, component_amount)[0])

        return orders

    def generate_orders_djembes(self, symbol: str, state: TradingState) -> List[Order]:
        orders = []

        if len(self.delta1_history) >= self.std_window:
            # 确保delta1_history不为空
            delta1 = self.delta1_history[-1] if self.delta1_history else 0
            std_dev1 = np.std(self.delta1_history)

            # 应用相关性过滤器
            should_trade, volume_factor, threshold_factor = self._apply_correlation_filter('PICNIC_BASKET1')

            if should_trade and abs(delta1) > self.std_multiplier * threshold_factor * std_dev1:
                basket_trade_direction = 1 if delta1 > 0 else -1
                base_amount = 1 * basket_trade_direction * self._get_max_possible_trade('PICNIC_BASKET1', state,
                                                                                        basket_trade_direction)
                component_amount = -int(base_amount * volume_factor)  # 应用量调整系数

                if component_amount != 0:
                    orders.extend(self.quick_trade(symbol, state, component_amount)[0])

        return orders

    def generate_orders(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        # 计算并记录当前delta
        delta1 = self.get_price_delta(state, 'PICNIC_BASKET1')
        delta2 = self.get_price_delta(state, 'PICNIC_BASKET2')

        # 只有当计算结果有效（非零）时才记录
        if delta1 != 0:
            self.delta1_history.append(delta1)
        if delta2 != 0:
            self.delta2_history.append(delta2)

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
            'basket_price_history': {
                basket: list(history) for basket, history in self.basket_price_history.items()
            },
            'basket_value_history': {
                basket: list(history) for basket, history in self.basket_value_history.items()
            },
            'ewma_correlation': self.ewma_correlation,
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

                # 加载价格和价值历史
                if 'basket_price_history' in basket_data:
                    for basket, history in basket_data['basket_price_history'].items():
                        self.basket_price_history[basket] = deque(history, maxlen=30)

                if 'basket_value_history' in basket_data:
                    for basket, history in basket_data['basket_value_history'].items():
                        self.basket_value_history[basket] = deque(history, maxlen=30)

                # 加载EWMA相关性
                if 'ewma_correlation' in basket_data:
                    self.ewma_correlation = basket_data['ewma_correlation']

                # 加载组件价格
                if 'component_prices' in basket_data:
                    self.component_prices = basket_data['component_prices']
            except Exception as e:
                logger.print(f"Error loading state: {str(e)}")

    def calculate_ewma_correlation(self, basket: str) -> float:
        """计算篮子价格与理论价值之间的EWMA相关性"""
        # 需要至少5个数据点才能计算有意义的相关性
        if len(self.basket_price_history[basket]) < 5 or len(self.basket_value_history[basket]) < 5:
            return self.ewma_correlation[basket]  # 返回当前值

        prices = np.array(list(self.basket_price_history[basket]))
        values = np.array(list(self.basket_value_history[basket]))

        # 计算价格和价值的变化率，避免绝对价格可能导致的虚假相关性
        price_changes = np.diff(prices) / prices[:-1]
        value_changes = np.diff(values) / values[:-1]

        # 确保有足够的变化率数据
        if len(price_changes) < 4 or len(value_changes) < 4:
            return self.ewma_correlation[basket]

        # 计算当前批次的相关系数
        current_corr = np.corrcoef(price_changes, value_changes)[0, 1]
        if np.isnan(current_corr):
            return self.ewma_correlation[basket]

        # 应用EWMA更新
        new_corr = self.ewma_alpha * current_corr + (1 - self.ewma_alpha) * self.ewma_correlation[basket]
        self.ewma_correlation[basket] = new_corr

        return new_corr


class VolcanicRockStrategy(Strategy):
    def __init__(self, symbols: List[str], position_limits: dict, time_window: int , iv_threshold: float):
        # 使用第一个symbol作为虚拟主产品
        super().__init__(symbols[0], position_limits[symbols[0]])
        self.symbols = symbols
        self.K = [int(symbol.split("_")[-1]) for symbol in symbols[1:]]
        self.position_limits = position_limits
        self.time_window = time_window
        self.iv_positive_threshold = iv_threshold
        self.iv_negative_threshold = -iv_threshold
        self.history = {}
        self.T = 5/365
        self.timestamp_high = 1000000
        self.timestamp_unit = 100
        self.betas = []
        self.raw_ivs = []
        self.ivs = []
        self.base_ivs = deque(maxlen = time_window)
        self.outlier_times = 0

        

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
            return price_sum, total_volume if total_volume > 0 else 0

        # 计算买卖方加权均价
        buy_sum, buy_volume = weighted_avg(order_depth.buy_orders, n=3)  # 买单簿是字典
        sell_sum, sell_volume = weighted_avg(order_depth.sell_orders, n=3)  # 卖单簿是字典
        if buy_volume + sell_volume == 0:
            return 0
        else:
            return (buy_sum + sell_sum) / (buy_volume + sell_volume)

    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def bs_call_price(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)  # 到期或波动率为0时的payoff

        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        call_price = S * self.norm_cdf(d1) - K * self.norm_cdf(d2)
        return call_price

    def implied_volatility_call_bisect(self, market_price, S, K, T, 
                                    sigma_low=0.001, sigma_high=0.8, 
                                    tol=1e-8, max_iter = 500):
        def objective(sigma):
            return self.bs_call_price(S, K, T, sigma) - market_price

        low = sigma_low
        high = sigma_high

        f_low = objective(low)
        f_high = objective(high)

        if f_low > 0:
            return sigma_low
        if f_high < 0:
            return sigma_high
        
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            f_mid = objective(mid)

            if abs(f_mid) < tol:
                return mid

            if f_low * f_mid < 0:
                high = mid
                f_high = f_mid
            else:
                low = mid
                f_low = f_mid

        return 0.5 * (low + high)
    

    @staticmethod
    def fit_vol_surface(m, iv):
        m = np.array(m)
        iv = np.array(iv)

        # 原始二次拟合
        X_full = np.column_stack([
            np.ones_like(m),    # β₀
            m,                  # β₁
            m**2,               # β₂
        ])
        beta = np.linalg.lstsq(X_full, iv, rcond=None)[0]

        # 如果开口向下，就 fallback 到无 β₁ 项的拟合
        if beta[2] < 0:
            # fallback: 只用 β₀ + β₂·m² 拟合，强制 β₂ ≥ 0
            X_restrict = np.column_stack([
                np.ones_like(m),
                m**2
            ])
            beta_restrict = np.linalg.lstsq(X_restrict, iv, rcond=None)[0]
            beta_0, beta_2 = beta_restrict

            # 强制 β₂ ≥ 0
            beta_2 = max(beta_2, 1e-6)

            beta = [beta_0, 0.0, beta_2]

        return list(beta)

    def compute_call_delta(self, S, K, T, sigma):
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        return self.norm_cdf(d1)
    
    def calculate_fair_value(self):
        """
        利用bs模型和波动率曲线拟合定价（含异常值处理和开口向上限制）
        """
        beta_update = True
        self.ivs = []
        self.raw_ivs = []

        for symbol in self.symbols:
            if symbol == "VOLCANIC_ROCK":
                current_S = self.history["VOLCANIC_ROCK"]['mid_price_history'][-1]
                continue

            K = int(symbol.split("_")[-1])
            data = self.history[symbol]
            voucher_price = data["mid_price_history"][-1]

            iv = self.implied_volatility_call_bisect(voucher_price, current_S, K, self.T)
            
            # 检查是否是边界异常值，决定是否更新 β
            if self.betas != [] and (iv == 0.001 or iv == 0.8):
                beta_update = False

            self.raw_ivs.append(iv)
            self.ivs.append(iv)

        # 计算 moneyness
        m = np.log(np.array(self.K) / current_S) / np.sqrt(self.T)
        m = np.array(m)
        ivs = np.array(self.ivs)

        # 过滤异常 iv：只保留合理范围内的点
        valid_mask = (ivs > 0.01) & (ivs < 0.5)
        m_filtered = m[valid_mask]
        ivs_filtered = ivs[valid_mask]

        if beta_update and len(m_filtered) >= 3:
            self.betas = self.fit_vol_surface(m_filtered, ivs_filtered)

        # 拟合 IV 曲线
        X = np.column_stack([
            np.ones_like(m),
            m,
            m**2
        ])
        iv_fitted = X @ self.betas

        # 限制拟合后的 IV 在合理范围
        iv_fitted = np.clip(iv_fitted, 0.01, 0.5)

        # 平移 base_iv 修正
        base_iv = self.betas[0]
        self.base_ivs.append(base_iv)

        scaled_rock_prices = (-5) * (self.history['VOLCANIC_ROCK']['mid_price_history'] / np.mean(self.history['VOLCANIC_ROCK']['mid_price_history']) - 1)
        scaled_base_ivs = (self.base_ivs / np.mean(self.base_ivs) - 1)

        base_iv_diff = scaled_base_ivs[-1] - scaled_rock_prices[-1]
        base_iv_diff = 0
        # 应用 base_iv_diff 修正
        self.ivs = list(iv_fitted - base_iv_diff)

        # 计算各期权 fair price
        fair_prices = [
            self.bs_call_price(current_S, self.K[i], self.T, sigma=self.ivs[i])
            for i in range(len(self.symbols) - 1)
        ]

        return fair_prices

    
    def calculate_current_portfolio_delta(self, state: TradingState) -> float:
        total_delta = 0.0
        S = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK"])
        if S is None:
            return 0.0

        for i in range(1, len(self.symbols)):
            position = state.position.get(self.symbols[i], 0)
            if position == 0:
                continue

            K = int(self.symbols[i].split("_")[-1])
            price = self.calculate_mid_price(state.order_depths[self.symbols[i]])
            sigma = self.ivs[i - 1]
            if sigma is None:
                continue

            delta = self.compute_call_delta(S, K, self.T, sigma)
            total_delta += delta * position

        return total_delta
  
    #下单函数

    def generate_rock_orders(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = []
        return orders
    
    def generate_orders_fair_value_arbitrage(self, state: TradingState) -> Tuple[Dict[str, List[Order]], float]:
        """
        根据fair price进行套利
        """
        orders = {}
        total_delta_exposure = 0.0  # 用局部变量，不要用 self 记录，避免多次调用累加错

        fair_prices = self.calculate_fair_value()


        for i in range(1, len(self.symbols)):
            order_depth = state.order_depths[self.symbols[i]]
            mid_price = self.calculate_mid_price(order_depth)
            fair_price = fair_prices[i - 1]
            position = state.position.get(self.symbols[i], 0)
            limit = self.position_limits[self.symbols[i]]

            delta_p = fair_price - mid_price
            orders_for_symbol = []

            # 获取 strike 和基础价格
            K = int(self.symbols[i].split("_")[-1])
            S = self.calculate_mid_price(state.order_depths["VOLCANIC_ROCK"])
            T = self.T
            
            sigma = self.ivs[i - 1]
            raw_iv = self.raw_ivs[i - 1]

            #logger.print(f"symbol: {self.symbols[i]}, delta_p: {delta_p}, delta_iv: {sigma - raw_iv}")
            
            # None check
            if mid_price is None or fair_price is None or S is None or sigma is None:
                continue

            option_delta = self.compute_call_delta(S, K, T, sigma)

            #利用iv来判断下单

            delta_iv = sigma - raw_iv

            #根据不同strick price 调整delta iv 阈值
            '''
            if K == 9500:
                self.iv_positive_threshold = 0.006
                self.iv_negative_threshold = -0.01

            if K == 9750:
                self.iv_positive_threshold = 0.025
                self.iv_negative_threshold = -0.015

            if K == 10000:
                self.iv_positive_threshold = 0.008
                self.iv_negative_threshold = -0.018
            
            if K == 10250:
                self.iv_positive_threshold = 0.01
                self.iv_negative_threshold = -0.01
            
            if K == 10500:
                self.iv_positive_threshold = 0.005
                self.iv_negative_threshold = -0.006
            '''
            if delta_iv > self.iv_positive_threshold:
                #市场iv偏低，买入
                for price, amount in order_depth.sell_orders.items():
                        if self.symbols[i] not in orders:
                            orders[self.symbols[i]] = []
                        amount = -min(-amount, limit - position)
                        #amount = int(min(abs(delta_iv - self.iv_positive_threshold)/ 0.5, 1) * amount)
                        orders[self.symbols[i]].append(Order(self.symbols[i], price,  -amount)) #买入
                        position += amount
                        total_delta_exposure += option_delta * (-amount)

            if delta_iv < self.iv_negative_threshold:
                #市场iv偏高，卖出
                for price, amount in order_depth.buy_orders.items(): 
                        if self.symbols[i] not in orders:
                            orders[self.symbols[i]] = []
                        amount = min(amount, position + limit)
                        #根据价差比例下单
                        #amount = int(min(abs(self.iv_negative_threshold - delta_iv)/ 0.5, 1) * amount)
                        orders[self.symbols[i]].append(Order(self.symbols[i], price,  -amount)) #卖出
                        position -= amount
                        total_delta_exposure += option_delta * amount * -1

            if orders_for_symbol:
                orders[self.symbols[i]] = orders_for_symbol

        return orders, total_delta_exposure
    
    def generate_orders_pair_arbitrage(self, state: TradingState, exist_orders: dict) -> Tuple[Dict[str, List[Order]], float]:
        orders = {}
        total_delta_exposure = 0.0

        current_positions = state.position.copy()
        ivs = self.raw_ivs  
        fitted_ivs = self.ivs 

        pairs_candidate = []

        for i in range(1, len(self.symbols)):
            for j in range(i + 1, len(self.symbols)):
                symbol_i = self.symbols[i]
                symbol_j = self.symbols[j]
                iv_i = ivs[i-1]
                iv_j = ivs[j-1]
                fitted_iv_i = fitted_ivs[i-1]
                fitted_iv_j = fitted_ivs[j-1]

                delta_iv_i = fitted_iv_i - iv_i
                delta_iv_j = fitted_iv_j - iv_j
                delta_diff = abs(delta_iv_i - delta_iv_j)

                if delta_diff > self.iv_positive_threshold:
                    pairs_candidate.append((symbol_i, symbol_j, delta_iv_i, delta_iv_j, delta_diff))

        for symbol_i, symbol_j, delta_iv_i, delta_iv_j, delta_diff in pairs_candidate:
            pos_i = current_positions.get(symbol_i, 0)
            pos_j = current_positions.get(symbol_j, 0)
            limit_i = self.position_limits.get(symbol_i, 0)
            limit_j = self.position_limits.get(symbol_j, 0)

            depth_i = state.order_depths[symbol_i]
            depth_j = state.order_depths[symbol_j]

            if symbol_i not in orders:
                orders[symbol_i] = []
            if symbol_j not in orders:
                orders[symbol_j] = []

            # 根据 IV 差方向判断买谁、卖谁（吃市场）
            if delta_iv_i < delta_iv_j:
                # 买 i 卖 j
                buy_symbol, sell_symbol = symbol_i, symbol_j
                buy_depth, sell_depth = depth_i, depth_j
                buy_iv, sell_iv = delta_iv_i, delta_iv_j
            else:
                # 买 j 卖 i
                buy_symbol, sell_symbol = symbol_j, symbol_i
                buy_depth, sell_depth = depth_j, depth_i
                buy_iv, sell_iv = delta_iv_j, delta_iv_i

            buy_orders = sorted(buy_depth.sell_orders.items())  # 买一方吃卖单
            sell_orders = sorted(sell_depth.buy_orders.items(), reverse=True)  # 卖一方吃买单

            buy_position = current_positions.get(buy_symbol, 0)
            sell_position = current_positions.get(sell_symbol, 0)
            buy_limit = self.position_limits.get(buy_symbol, 0)
            sell_limit = self.position_limits.get(sell_symbol, 0)

            max_qty = min(buy_limit - buy_position, sell_limit - sell_position, int(delta_diff * 100))
            filled_qty = 0

            for (buy_price, buy_amt), (sell_price, sell_amt) in zip(buy_orders, sell_orders):
                trade_qty = min(-buy_amt, sell_amt, max_qty - filled_qty)
                if trade_qty <= 0:
                    continue

                orders[buy_symbol].append(Order(buy_symbol, buy_price, trade_qty))
                orders[sell_symbol].append(Order(sell_symbol, sell_price, -trade_qty))

                total_delta_exposure += buy_iv * trade_qty + sell_iv * trade_qty
                filled_qty += trade_qty

                if filled_qty >= max_qty:
                    break

            # 合并已有订单
            if symbol_i in exist_orders:
                orders[symbol_i].extend(exist_orders[symbol_i])
            if symbol_j in exist_orders:
                orders[symbol_j].extend(exist_orders[symbol_j])

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
        #current_delta = self.calculate_current_portfolio_delta(state)

        arbitrage_orders, total_delta_exposure = self.generate_orders_fair_value_arbitrage(state)

        #arbitrage_orders, total_delta_exposure = self.generate_orders_pair_arbitrage(state, exist_orders=arbitrage_orders)

        #hedge_orders = self.generate_orders_delta_hedge(state, total_delta_exposure)

        rock_orders = self.generate_rock_orders(state)

        orders.update(arbitrage_orders)
        if rock_orders:
            if "VOLCANIC_ROCK" in rock_orders:
                orders["VOLCANIC_ROCK"] = rock_orders['VOLCANIC_ROCK']


        return orders
    

    def load_state(self, state):
        self.T -= self.timestamp_unit/self.timestamp_high/365
        logger.print("self.T: ", self.T)
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
                
        if len(self.base_ivs) > self.time_window:
            self.base_ivs.popleft()

    def save_state(self, state):
        pass

class MagnificentMacaronsStrategy(Strategy):
    def __init__(self, symbol, position_limit):
        super().__init__(symbol, position_limit)
    
    def generate_orders(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = []
        return []
    
    def load_state(self, state):
        observations = state.observations
        pass

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
                "time_window": 100,
                "iv_threshold": 0.012
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
