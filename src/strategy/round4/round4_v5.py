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
    def __init__(self, symbol: str, position_limit: int, ma_window: int = 200,
                 max_deviation: int = 200, band_width: float = 30, take_spread: float = 10, break_step: float = 15):
        super().__init__(symbol, position_limit)

        self.timestamp = 0
        self.open_price = 1931
        self.direction = 0

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
        best_bid_amount = order_depth.buy_orders[best_bid]
        best_ask_amount = order_depth.sell_orders[best_ask]

        fair_value = self.calculate_fair_value(state.order_depths[self.symbol])

        if fair_value > self.open_price + 50:
            available_sell = 50 + current_position
            if available_sell > 0:
                sell_amt = min(-best_ask_amount, available_sell)
                orders.append(Order(self.symbol, best_bid, -math.floor(sell_amt)))

        elif fair_value < self.open_price - 40:
            available_buy = 50 - current_position
            if available_buy > 0:
                buy_amt = min(best_bid_amount, available_buy)
                orders.append(Order(self.symbol, best_ask, math.floor(buy_amt)))

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
        stop_loss_threshold = entry_diff*1.6  # 例如，价差翻倍则止损

        if (current_diff < 0 and current_diff < stop_loss_threshold) or (
                current_diff > 0 and current_diff > stop_loss_threshold):
            logger.print(f"------------------{basket} current_diff {current_diff:.2f}---entry_diff--- {entry_diff:.2f}")
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

        # 参数设置
        min_trade_threshold = 15  # 只有当价差足够大时才交易
        max_component_units = 3  # 每次最多交易3个单位
        delta_list = [("PICNIC_BASKET1", self.delta1_history, 6),
                      ("PICNIC_BASKET2", self.delta2_history, 4)]

        for basket_name, delta_history, ratio in delta_list:
            if len(delta_history) < self.std_window:
                continue

            delta = delta_history[-1]
            std_dev = np.std(delta_history)

            # 确保波动正常，避免初期误判
            if std_dev < self.min_std_threshold:
                continue

            # 应用相关性过滤
            should_trade, volume_factor, threshold_factor = self._apply_correlation_filter(basket_name)
            if not should_trade:
                continue

            # 是否满足交易门槛
            adjusted_threshold = self.std_multiplier * threshold_factor * std_dev
            if abs(delta) < max(min_trade_threshold, adjusted_threshold):
                continue

            # 确定交易方向（篮子贵 -> 组件应买入）
            basket_dir = 1 if delta > 0 else -1
            base_basket_units = self._get_max_possible_trade(basket_name, state, basket_dir)
            base_component_units = -int(ratio * base_basket_units * volume_factor)

            # 限制最大交易单位（保守）
            component_units = max(-max_component_units, min(base_component_units, max_component_units))
            if component_units != 0:
                trades, _ = self.quick_trade(symbol, state, component_units)
                orders.extend(trades)

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

            if should_trade and abs(delta1) > 10:
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

            if abs(delta1) > self.std_multiplier * threshold_factor * std_dev1:
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
        # 如果没有足够的数据，直接返回当前值
        if (basket not in self.basket_price_history or basket not in self.basket_value_history or
                len(self.basket_price_history[basket]) < 5 or len(self.basket_value_history[basket]) < 5):
            return self.ewma_correlation.get(basket, 0.8)  # 使用默认值或当前值

        try:
            # 转换为numpy数组
            prices = np.array(list(self.basket_price_history[basket]), dtype=np.float64)
            values = np.array(list(self.basket_value_history[basket]), dtype=np.float64)

            # 检查数据有效性
            if (len(prices) != len(values) or
                    np.any(~np.isfinite(prices)) or np.any(~np.isfinite(values)) or  # 检查无穷和NaN
                    np.any(prices <= 0) or np.any(values <= 0)):  # 确保所有值都是正数
                logger.print(f"Invalid data detected in correlation calculation for {basket}")
                return self.ewma_correlation.get(basket, 0.8)

            # 计算对数收益率而不是百分比变化，避免极端值
            price_returns = np.log(prices[1:]) - np.log(prices[:-1])
            value_returns = np.log(values[1:]) - np.log(values[:-1])

            # 检查计算结果
            if (len(price_returns) < 4 or len(value_returns) < 4 or
                    np.any(~np.isfinite(price_returns)) or np.any(~np.isfinite(value_returns))):
                logger.print(f"Invalid returns detected in correlation calculation for {basket}")
                return self.ewma_correlation.get(basket, 0.8)

            # 使用masked array处理可能的异常值
            mask = np.abs(price_returns) > 0.2  # 过滤掉超过20%的收益率变化
            price_returns = np.ma.array(price_returns, mask=mask)
            value_returns = np.ma.array(value_returns, mask=mask)

            if price_returns.count() < 4 or value_returns.count() < 4:
                logger.print(f"Not enough valid data points after filtering for {basket}")
                return self.ewma_correlation.get(basket, 0.8)

            # 手动计算相关系数，避免使用np.corrcoef
            p_mean = price_returns.mean()
            v_mean = value_returns.mean()
            p_std = price_returns.std()
            v_std = value_returns.std()

            # 检查标准差是否为零
            if p_std == 0 or v_std == 0:
                logger.print(f"Zero standard deviation detected for {basket}")
                return self.ewma_correlation.get(basket, 0.8)

            # 计算协方差
            cov = ((price_returns - p_mean) * (value_returns - v_mean)).mean()

            # 计算相关系数
            current_corr = cov / (p_std * v_std)

            # 检查结果是否是有效的相关系数
            if not (-1.0 <= current_corr <= 1.0) or np.isnan(current_corr):
                logger.print(f"Invalid correlation coefficient: {current_corr} for {basket}")
                return self.ewma_correlation.get(basket, 0.8)

            # 应用EWMA更新
            new_corr = self.ewma_alpha * current_corr + (1 - self.ewma_alpha) * self.ewma_correlation.get(basket, 0.8)
            self.ewma_correlation[basket] = new_corr

            return new_corr

        except Exception as e:
            # 捕获所有异常
            logger.print(f"Error in calculating correlation for {basket}: {str(e)}")
            return self.ewma_correlation.get(basket, 0.8)


class VolcanicRockStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int):
        super().__init__(symbol, position_limit)

        self.symbol = symbol
        self.position_limit = position_limit
        self.active_atm = None
        self.current_mode = "Normal"
        self.active_time_value = 0
        self.rock_fair_value_history = deque(maxlen=80)
        self.ma_80 = 0
        self.time_counter = 0
        self.active_direction = 0
        self.up_close_value = 118
        self.down_close_value = 77
        self.voucher_config = {
            "VOLCANIC_ROCK_VOUCHER_9500": {
                "strike": 9500,
                "position_limit": 200,
                "lower_bound": None,
                "upper_bound": 9550,
            },
            "VOLCANIC_ROCK_VOUCHER_9750": {
                "strike": 9750,
                "position_limit": 200,
                "lower_bound": 9550,
                "upper_bound": 9800,
            },
            "VOLCANIC_ROCK_VOUCHER_10000": {
                "strike": 10000,
                "position_limit": 200,
                "lower_bound": 9800,
                "upper_bound": 10050,

            },
            "VOLCANIC_ROCK_VOUCHER_10250": {
                "strike": 10250,
                "position_limit": 200,
                "lower_bound": 10050,
                "upper_bound": 10300,

            },
            "VOLCANIC_ROCK_VOUCHER_10500": {
                "strike": 10500,
                "position_limit": 200,
                "lower_bound": 10300,
                "upper_bound": None
            },
        }

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
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
        if buy_avg and sell_avg:
            return (buy_avg + sell_avg) / 2
        elif buy_avg:
            return buy_avg
        elif sell_avg:
            return sell_avg

    def calculate_time_value(self, voucher_symbol: str, ma_80, voucher_price: float) -> float:
        config = self.voucher_config.get(voucher_symbol)
        return config["strike"] + voucher_price - ma_80 if config else 0.0

    def determine_ATM(self, ma_100) -> str:
        if ma_100 >= 10300:
            return "VOLCANIC_ROCK_VOUCHER_10500"
        elif ma_100 >= 10050:
            return "VOLCANIC_ROCK_VOUCHER_10250"
        elif ma_100 >= 9800:
            return "VOLCANIC_ROCK_VOUCHER_10000"
        elif ma_100 >= 9550:
            return "VOLCANIC_ROCK_VOUCHER_9750"
        else:
            return "VOLCANIC_ROCK_VOUCHER_9500"

    def generate_orders(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        rock_position = state.position.get("VOLCANIC_ROCK", 0)

        # 获取rock资产价格
        rock_order_depth = state.order_depths.get("VOLCANIC_ROCK", OrderDepth())
        S = self.calculate_mid_price(rock_order_depth)
        if S == 0:
            return orders

        # 添加rock_fair_value_history
        self.rock_fair_value_history.append(S)
        if len(self.rock_fair_value_history) > 80:
            self.rock_fair_value_history.popleft()
        if len(self.rock_fair_value_history) >= 80:
            self.ma_80 = np.mean(list(self.rock_fair_value_history)[-80:])
        else:
            self.ma_80 = S
            return orders

        # 确定当前ATM
        current_atm = self.determine_ATM(self.ma_80)
        atm_order_depth = state.order_depths.get(current_atm, OrderDepth())
        atm_price = self.calculate_mid_price(atm_order_depth)
        if atm_price == 0:
            return orders
        time_value = self.calculate_time_value(current_atm, self.ma_80, atm_price)

        for symbol in self.voucher_config:
            if self.voucher_config[symbol]["strike"] > S + 200:
                depth = state.order_depths.get(symbol, OrderDepth())
                if not depth.buy_orders:
                    continue
                best_ask = min(depth.sell_orders.keys())
                ask_vol = depth.sell_orders[best_ask]
                current_pos = state.position.get(symbol, 0)
                available_buy = 200 - current_pos
                buy_quan = min(available_buy, -ask_vol)
                if available_buy > 0 and best_ask < 5:
                    if symbol not in orders:
                        orders[symbol] = []
                    orders[symbol].append(Order(symbol, best_ask, round(buy_quan)))

        # 如果当前是Normal模式
        if self.current_mode == "Normal":
            if time_value < 98:
                self.current_mode = "Abnormal"
                self.active_atm = current_atm
                self.active_time_value = time_value
                self.active_direction = -1
                # 卖出标的资产
                qty = self.position_limit + rock_position
                if qty > 0 and rock_order_depth.buy_orders:
                    best_bid = min(rock_order_depth.buy_orders.keys())
                    best_bid_amount = rock_order_depth.buy_orders[best_bid]
                    sell_qty = min(50, best_bid_amount, qty)
                    orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -round(sell_qty))]
                # 卖出低行权价凭证
                active_strike = self.voucher_config[current_atm]["strike"]
                for symbol in self.voucher_config:
                    if self.voucher_config[symbol]["strike"] <= active_strike:
                        depth = state.order_depths.get(symbol, OrderDepth())
                        current_pos = state.position.get(symbol, 0)
                        available_sell = 200 + current_pos
                        if available_sell > 0 and depth.buy_orders:
                            # 按买单价降序处理所有档位
                            sorted_bids = sorted(depth.buy_orders.keys(), reverse=True)
                            for bid_price in sorted_bids:
                                if available_sell <= 0:
                                    break
                                bid_vol = depth.buy_orders[bid_price]
                                fillable = min(bid_vol, available_sell)
                                orders.setdefault(symbol, []).append(Order(symbol, bid_price, -math.floor(fillable)))
                                available_sell -= fillable
                return orders
            elif time_value > 142:
                self.current_mode = "Abnormal"
                self.active_atm = current_atm
                self.active_time_value = time_value
                self.active_direction = 1
                # 买入标的资产
                qty = self.position_limit - rock_position
                if qty > 0 and rock_order_depth.sell_orders:
                    best_ask = min(rock_order_depth.sell_orders.keys())
                    best_ask_amount = rock_order_depth.sell_orders[best_ask]
                    buy_qty = min(50, -best_ask_amount, qty)
                    orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_ask, round(buy_qty))]
                # 买入低行权价凭证
                active_strike = self.voucher_config[current_atm]["strike"]
                for symbol in self.voucher_config:
                    if self.voucher_config[symbol]["strike"] <= active_strike:
                        depth = state.order_depths.get(symbol, OrderDepth())
                        current_pos = state.position.get(symbol, 0)
                        available_buy = 200 - current_pos

                        if available_buy > 0 and depth.sell_orders:
                            # 按卖单价降序处理所有档位
                            sorted_ask = sorted(depth.sell_orders.keys())
                            for ask_price in sorted_ask:
                                if available_buy <= 0:
                                    break
                                ask_vol = depth.sell_orders[ask_price]
                                fillable = min(-ask_vol, available_buy)
                                orders.setdefault(symbol, []).append(Order(symbol, ask_price, math.floor(fillable)))
                                available_buy -= fillable
                return orders

        # 如果当前是Abnormal模式
        elif self.current_mode == "Abnormal":
            # 如果当前rock还在active_atm区间
            if self.active_atm == current_atm and self.active_direction == -1:
                atm_depth = state.order_depths.get(current_atm, OrderDepth())
                atm_price = self.calculate_mid_price(atm_depth)
                current_tv = self.calculate_time_value(current_atm, self.ma_80, atm_price)
                # 继续卖出标的
                if current_tv < self.active_time_value:
                    self.active_time_value = current_tv
                    available_sell = 400 + rock_position
                    if available_sell > 0 and rock_order_depth.buy_orders:
                        best_bid = min(rock_order_depth.buy_orders.keys())
                        best_bid_amount = rock_order_depth.buy_orders[best_bid]
                        sell_qty = min(50, best_bid_amount, math.floor(0.3 * available_sell))
                        orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -math.floor(sell_qty))]

                    for symbol in self.voucher_config:
                        active_strike = self.voucher_config[current_atm]["strike"]
                        if self.voucher_config[symbol]["strike"] <= active_strike:
                            depth = state.order_depths.get(symbol, OrderDepth())
                            if not depth.buy_orders:
                                continue
                            best_bid = min(depth.buy_orders.keys())
                            best_bid_amount = depth.buy_orders[best_bid]
                            current_pos = state.position.get(symbol, 0)
                            available_sell = 200 + current_pos
                            sell_qty = min(best_bid_amount, math.floor(0.3 * available_sell))
                            if sell_qty > 0:
                                if symbol not in orders:
                                    orders[symbol] = []
                                orders[symbol].append(Order(symbol, best_bid, -sell_qty))
                    return orders
                # 平仓逻辑
                elif current_tv > self.up_close_value:
                    # 买入平仓标的资产
                    total_position = rock_position  # 初始化总持仓计算
                    if rock_position < 0:
                        if rock_order_depth.sell_orders:
                            best_ask = max(rock_order_depth.sell_orders.keys())
                            best_ask_amount = rock_order_depth.sell_orders[best_ask]
                            available_buy = 400 - rock_position
                            if available_buy > 0:
                                buy_qty = min(50, -best_ask_amount, math.floor(0.3 * available_buy))
                                orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_ask, round(buy_qty))]
                    for symbol in self.voucher_config:
                        active_strike = self.voucher_config[current_atm]["strike"]
                        if self.voucher_config[symbol]["strike"] <= active_strike + 300:
                            current_pos = state.position.get(symbol, 0)
                            if current_pos >= 0:
                                continue
                            else:
                                total_position += current_pos
                            depth = state.order_depths.get(symbol, OrderDepth())
                            if not depth.sell_orders:
                                continue  # 需要处理无买单情况
                            best_ask = max(depth.sell_orders.keys())
                            best_ask_amount = depth.sell_orders[best_ask]
                            available_buy = 200 - current_pos  # 正确平仓数量
                            if symbol not in orders:
                                orders[symbol] = []
                            if available_buy > 0:
                                buy_qty = min(-best_ask_amount, math.floor(0.3 * available_buy))
                                orders[symbol].append(Order(symbol, best_ask, buy_qty))

                    # 严格检查所有仓位
                    if total_position >= -40:
                        self.current_mode = "Normal"
                        self.active_atm = None
                        self.time_counter = 0
                        self.active_time_value = 0
                        self.active_direction = 0
                    return orders

            elif self.active_atm == current_atm and self.active_direction == 1:
                atm_depth = state.order_depths.get(current_atm, OrderDepth())
                atm_price = self.calculate_mid_price(atm_depth)
                current_tv = self.calculate_time_value(current_atm, self.ma_80, atm_price)
                # 继续买入标的
                if current_tv > self.active_time_value:
                    self.active_time_value = current_tv
                    availeble_buy = 400 - rock_position
                    if availeble_buy > 0 and rock_order_depth.sell_orders:
                        best_ask = min(rock_order_depth.sell_orders.keys())
                        best_ask_amount = rock_order_depth.sell_orders[best_ask]
                        buy_qty = min(50, best_ask_amount, math.floor(0.3 * availeble_buy))
                        orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_ask, buy_qty)]
                    for symbol in self.voucher_config:
                        active_strike = self.voucher_config[current_atm]["strike"]
                        if self.voucher_config[symbol]["strike"] <= active_strike:
                            depth = state.order_depths.get(symbol, OrderDepth())
                            if not depth.sell_orders:
                                continue
                            best_ask = min(depth.sell_orders.keys())
                            best_ask_amount = depth.sell_orders[best_ask]
                            current_pos = state.position.get(symbol, 0)
                            available_buy = 200 - current_pos
                            buy_qty = min(-best_ask_amount, math.floor(0.3 * available_buy))
                            if symbol not in orders:
                                orders[symbol] = []
                            orders[symbol].append(Order(symbol, best_ask, buy_qty))
                    return orders
                # 平仓逻辑
                elif current_tv < self.down_close_value:
                    # 卖出平仓标的资产
                    total_position = rock_position  # 初始化总持仓计算
                    if rock_position > 0:
                        if rock_order_depth.buy_orders:
                            best_bid = max(rock_order_depth.buy_orders.keys())
                            best_bid_amount = rock_order_depth.buy_orders[best_bid]
                            available_sell = self.position_limit + rock_position
                            if available_sell > 0:
                                sell_qty = min(50, best_bid_amount, math.floor(0.3 * available_sell))
                                orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -sell_qty)]
                    for symbol in self.voucher_config:
                        active_strike = self.voucher_config[current_atm]["strike"]
                        if self.voucher_config[symbol]["strike"] <= active_strike + 300:
                            current_pos = state.position.get(symbol, 0)
                            if current_pos <= 0:
                                continue
                            else:
                                total_position += current_pos
                            depth = state.order_depths.get(symbol, OrderDepth())
                            if not depth.buy_orders:
                                continue  # 需要处理无买单情况
                            best_bid = max(depth.buy_orders.keys())
                            best_bid_amount = depth.buy_orders[best_bid]
                            available_sell = 200 + current_pos  # 正确平仓数量
                            if symbol not in orders:
                                orders[symbol] = []
                            if available_sell > 0:
                                sell_qty = min(best_bid_amount, math.floor(0.3 * available_sell))
                                orders[symbol].append(Order(symbol, best_bid, -sell_qty))

                    # 严格检查所有仓位
                    if total_position <= 40:
                        self.current_mode = "Normal"
                        self.active_atm = None
                        self.time_counter = 0
                        self.active_time_value = 0
                        self.active_direction = 0
                    return orders

            elif self.active_atm != current_atm and self.active_atm is not None and self.active_direction == -1:
                if self.voucher_config[current_atm]["strike"] < self.voucher_config[self.active_atm]["strike"]:
                    self.active_atm = current_atm

                elif self.voucher_config[current_atm]["strike"] > self.voucher_config[self.active_atm]["strike"]:
                    self.time_counter += 1
                    total_position = rock_position if rock_position < 0 else 0  # 初始化总持仓计算
                    # 止损标的资产
                    if rock_position < 0:
                        if rock_order_depth.sell_orders:
                            best_ask = max(rock_order_depth.sell_orders.keys())
                            best_ask_amount = rock_order_depth.sell_orders[best_ask]
                            bound = self.voucher_config[current_atm]["lower_bound"]
                            available_buy = self.position_limit - rock_position
                            if available_buy > 0 and best_ask <= bound + 20:
                                buy_qty = min(50, -best_ask_amount, math.floor(0.3 * available_buy))
                                orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_ask, buy_qty)]
                                # 止损所有凭证
                                for symbol in self.voucher_config:
                                    active_strike = self.voucher_config[current_atm]["strike"]
                                    if self.voucher_config[symbol]["strike"] <= active_strike + 300:
                                        current_pos = state.position.get(symbol, 0)
                                        if current_pos < 0:
                                            total_position += current_pos
                                        elif current_pos >= 0:
                                            continue
                                        depth = state.order_depths.get(symbol, OrderDepth())
                                        if not depth.sell_orders:
                                            continue  # 需要处理无买单情况
                                        best_ask = max(depth.sell_orders.keys())
                                        best_ask_amount = depth.sell_orders[best_ask]
                                        available_buy = 200 - current_pos  # 平仓数量
                                        if symbol not in orders:
                                            orders[symbol] = []
                                        if available_buy > 0:
                                            buy_qty = min(-best_ask_amount, math.floor(0.3 * available_buy))
                                            orders[symbol].append(Order(symbol, best_ask, buy_qty))

                    # 严格检查所有仓位
                    if total_position >= -40 or self.time_counter == 15:
                        self.current_mode = "Normal"
                        self.active_atm = None
                        self.active_time_value = 0
                        self.time_counter = 0
                        self.active_direction = 0
                    return orders

            elif self.active_atm != current_atm and self.active_atm is not None and self.active_direction == 1:
                if self.voucher_config[current_atm]["strike"] > self.voucher_config[self.active_atm]["strike"]:
                    self.active_atm = current_atm

                elif self.voucher_config[current_atm]["strike"] < self.voucher_config[self.active_atm]["strike"]:
                    self.time_counter += 1
                    total_position = rock_position if rock_position > 0 else 0  # 初始化总持仓计算
                    # 止损标的资产
                    if rock_position > 0:
                        if rock_order_depth.buy_orders:
                            best_bid = max(rock_order_depth.buy_orders.keys())
                            best_bid_amount = rock_order_depth.buy_orders[best_bid]
                            bound = self.voucher_config[current_atm]["upper_bound"]
                            available_sell = self.position_limit + rock_position
                            if available_sell > 0 and best_bid >= bound - 20:
                                sell_qty = min(50, best_bid_amount, math.floor(0.3 * available_sell))
                                orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", best_bid, -sell_qty)]
                                # 止损所有凭证
                                for symbol in self.voucher_config:
                                    active_strike = self.voucher_config[current_atm]["strike"]
                                    if self.voucher_config[symbol]["strike"] <= active_strike + 300:
                                        current_pos = state.position.get(symbol, 0)
                                        if current_pos > 0:
                                            total_position += current_pos
                                        elif current_pos <= 0:
                                            continue
                                        depth = state.order_depths.get(symbol, OrderDepth())
                                        if not depth.buy_orders:
                                            continue  # 需要处理无买单情况
                                        best_bid = max(depth.buy_orders.keys())
                                        best_bid_amount = depth.buy_orders[best_bid]
                                        available_sell = 200 + current_pos  # 正确平仓数量
                                        if symbol not in orders:
                                            orders[symbol] = []
                                        if available_sell > 0:
                                            sell_qty = min(best_bid_amount, math.floor(0.3 * available_sell))
                                            orders[symbol].append(Order(symbol, best_bid, -sell_qty))

                    # 严格检查所有仓位
                    if total_position <= 40 or self.time_counter == 20:
                        self.current_mode = "Normal"
                        self.active_atm = None
                        self.active_time_value = 0
                        self.time_counter = 0
                        self.active_direction = 0

                    return orders
        return orders

    def save_state(self, state):
        return {}

    def load_state(self, state):
        pass

class MacaronStrategy(Strategy):
    """马卡龙交易策略"""

    def __init__(self,
                 symbol: str,
                 position_limit: int,
                 conversion_limit: int = 10,
                 sun_threshold: int = 60,
                 sun_coeff: float = 0.10,
                 sugar_threshold: int = 40,
                 sugar_coeff: float = 0.08):
        super().__init__(symbol, position_limit)
        self.conversion_limit = conversion_limit
        self.storage_cost = 0.1
        self.current_mode = "Normal"

        self.sun_threshold = sun_threshold
        self.sun_coeff = sun_coeff
        self.sugar_threshold = sugar_threshold
        self.sugar_coeff = sugar_coeff

        # 历史数据
        self.fair_value_history = deque(maxlen=100)
        self.sugar_price_history = deque(maxlen=100)
        self.sun_price_history = deque(maxlen=100)

        self.observation_history = deque(maxlen=100)
        self.position_history = deque(maxlen=100)
        self.conversion_count = 0
        self.position_entries = {}

        self.sunlightIndex_slope = 0
        self.sunlightIndex_history = deque(maxlen = 100)
        
        #计算持仓成本数据
        self.orders_history = {
            'long': [], 
            'short': []
        }

    def get_available_amount(self, symbol: str, state: TradingState) -> int:
        """
        返回市场上已有市价单的总数量
        sell_amount, buy_amount（注意都为正数）
        """
        order_depth = state.order_depths[symbol]
        sell_amount = -sum(order_depth.sell_orders.values())
        buy_amount = sum(order_depth.buy_orders.values())
        return sell_amount, buy_amount
    
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
                max_amount = min(-sell_amount, self.position_limit - position, amount)
                if max_amount > 0:
                    orders.append(Order(symbol, price, max_amount))
                    position += max_amount
                    amount -= max_amount
                if amount == 0:
                    break

        elif amount < 0:
            for price, buy_amount in sorted(order_depth.buy_orders.items()):
                max_amount = min(buy_amount, position + self.position_limit, -amount)  # amount已经是负数，卖出
                if max_amount > 0:
                    # 卖出
                    orders.append(Order(symbol, price, -max_amount))
                    position -= max_amount
                    amount += max_amount

                if amount == 0:
                    break

        return orders, amount
    
    def calculate_avg_cost(self, state: TradingState):
        """
        实时从 own_trades 中读取成交信息，计算平均持仓成本与当前净持仓
        自动处理多空平仓冲销逻辑
        """
        if not hasattr(self, "processed_trades"):
            self.processed_trades = set()
            self.orders_history = {
                'long': [],
                'short': []
            }
        def offset_inventory(order_list, amount):
            while amount > 0 and order_list:
                price, qty = order_list[0]
                if qty <= amount:
                    amount -= qty
                    order_list.pop(0)
                else:
                    order_list[0] = (price, qty - amount)
                    amount = 0
            return amount
        
        own_trades = state.own_trades.get(self.symbol, [])
        for trade in own_trades:
            trade_id = (trade.seller, trade.buyer, trade.price, trade.quantity, trade.timestamp)
            if trade_id in self.processed_trades:
                continue
            self.processed_trades.add(trade_id)
            

            # 判断你是买还是卖
            if trade.buyer == "SUBMISSION":
                direction = 1
            elif trade.seller == "SUBMISSION":
                direction = -1
            else:
                continue  # 不属于你自己的成交，跳过

            price = trade.price
            qty = trade.quantity

            # 平仓处理逻辑（offset 对手方向）
            if direction == 1:
                # 如果有空头仓位，优先冲销
                qty = offset_inventory(self.orders_history['short'], qty)
                if qty > 0:
                    self.orders_history['long'].append((price, qty))
            else:
                # 有多头仓位时优先冲销
                qty = offset_inventory(self.orders_history['long'], qty)
                if qty > 0:
                    self.orders_history['short'].append((price, qty))

        # 计算净仓与平均成本
        long_qty = sum(q for _, q in self.orders_history['long'])
        short_qty = sum(q for _, q in self.orders_history['short'])
        net_pos = long_qty - short_qty

        if net_pos > 0:
            total_cost = sum(price * qty for price, qty in self.orders_history['long'])
            avg_cost = total_cost / long_qty if long_qty else 0.0
        elif net_pos < 0:
            total_cost = sum(price * qty for price, qty in self.orders_history['short'])
            avg_cost = total_cost / short_qty if short_qty else 0.0
        else:
            avg_cost = 0.0

        return avg_cost, net_pos

    def update_storage_cost(self, state: TradingState):
        current_time = state.timestamp
        # 清理过期持仓记录（假设每个时间戳间隔固定）
        expired = [t for t in self.position_entries if current_time - t > 100]  # 假设100时间戳为周期
        for t in expired:
            del self.position_entries[t]
        # 记录当前持仓
        self.position_entries[current_time] = state.position.get(self.symbol, 0)
        # 计算总仓储成本
        total_storage_cost = 0.1 * sum(pos for pos in self.position_entries.values())
        return total_storage_cost

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        """计算市场公允价值"""

        def weighted_avg(prices_vols, n=3):
            """计算加权平均价格"""
            if not prices_vols:
                return 0

            is_buy = isinstance(prices_vols, dict) and len(prices_vols) > 0 and next(iter(prices_vols.values())) > 0
            sorted_orders = sorted(prices_vols.items(), key=lambda x: x[0], reverse=is_buy)[:n]

            if not sorted_orders:
                return 0

            total_vol = sum(abs(vol) for _, vol in sorted_orders)
            if total_vol == 0:
                return 0

            weighted_sum = sum(price * abs(vol) for price, vol in sorted_orders)
            return weighted_sum / total_vol

        buy_avg = weighted_avg(order_depth.buy_orders)
        sell_avg = weighted_avg(order_depth.sell_orders)
        if buy_avg > 0 and sell_avg > 0:
            return (buy_avg + sell_avg) / 2
        elif buy_avg > 0:
            return buy_avg
        elif sell_avg > 0:
            return sell_avg
        return 0

    def should_convert(self, state: TradingState, conversion_type: str) -> bool:
        """基于 avg_cost 判断是否可以通过 conversion 跨岛套利"""
        if self.conversion_count >= self.conversion_limit:
            return False

        if not state.observations or "MAGNIFICENT_MACARONS" not in state.observations.conversionObservations:
            return False

        obs = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
        position = state.position.get(self.symbol, 0)

        avg_cost, net_pos = self.calculate_avg_cost(state)

        # conversion 买入的成本
        buy_cost = obs.askPrice + obs.transportFees + obs.importTariff
        # conversion 卖出的收益
        sell_revenue = obs.bidPrice - obs.transportFees - obs.exportTariff

        min_profit = 0.1 # 利润门槛

        if conversion_type == "BUY" and position < self.position_limit:
            # 假设当前是空头，可以通过 conversion 买入来平空，或者反手做多
            # 判断当前持仓成本是否远高于跨岛买入
            return avg_cost > buy_cost + min_profit

        elif conversion_type == "SELL" and position > -self.position_limit:
            # 假设当前是多头，可以通过 conversion 卖出
            return sell_revenue - avg_cost > min_profit

        return False

    def process_market_data(self, state: TradingState):
        """处理市场数据并更新策略状态"""
        position = state.position.get(self.symbol, 0)
        self.position_history.append(position)
        
        # 计算并记录当前公允价值
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        fair_value = self.calculate_fair_value(order_depth)
        if fair_value > 0:
            self.fair_value_history.append(fair_value)
        # 记录行情观察值
        if state.observations and "MAGNIFICENT_MACARONS" in state.observations.conversionObservations:
            self.observation_history.append(state.observations.conversionObservations["MAGNIFICENT_MACARONS"])
        else: #如果没有数据，直接跳过
            return
        #记录糖价

        self.sugar_price_history.append(state.observations.conversionObservations["MAGNIFICENT_MACARONS"].sugarPrice)
        
        if state.observations.conversionObservations["MAGNIFICENT_MACARONS"].sunlightIndex is not None:
            sunlightIndex = state.observations.conversionObservations["MAGNIFICENT_MACARONS"].sunlightIndex
            self.sunlightIndex_history.append(sunlightIndex)
        else:
            sunlightIndex = None
        if sunlightIndex is not None:
            #根据SI或者相关系数设置突破条件
            if sunlightIndex < 45 and self.current_mode == 'Normal' and self.sunlightIndex_slope < 0:
                self.current_mode = "CSI"

        if len(self.sunlightIndex_history) > 5:
            self.sunlightIndex_slope = (self.sunlightIndex_history[-1] - self.sunlightIndex_history[-5]) / 5
        else:
            self.sunlightIndex_slope = 0


    def determine_optimal_conversions(self, state: TradingState) -> int:
        """基于 avg_cost 判断并决定 conversion 数量"""
        if not state.observations or "MAGNIFICENT_MACARONS" not in state.observations.conversionObservations:
            return 0

        position = state.position.get(self.symbol, 0)
        obs = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
        avg_cost, net_pos = self.calculate_avg_cost(state)

        conversions = 0
        min_profit = 0.1  # 可以调大点防止手续费侵蚀

        buy_cost = obs.askPrice + obs.transportFees + obs.importTariff
        sell_revenue = obs.bidPrice - obs.transportFees - obs.exportTariff

        # 如果当前是空头或无仓位，可以从岛外买入（conversion）补仓/反手
        if self.should_convert(state, "BUY"):
            profit = avg_cost - buy_cost
            if profit > min_profit :
                max_buy = min(self.conversion_limit - self.conversion_count, self.position_limit - position)
                conversions = max_buy

        # 如果当前是多头，可以把仓位卖给岛外
        elif self.should_convert(state, "SELL"):
            profit = sell_revenue - avg_cost
            if profit > min_profit:
                max_sell = min(self.conversion_limit - self.conversion_count, net_pos)
                conversions = -max_sell

        return conversions

    def generate_orders(self, state: TradingState) -> List[Order]:
        """生成订单和转换请求"""
        self.process_market_data(state)
        avg_cost, net_pos = self.calculate_avg_cost(state)
        orders = []
        conversions = 0
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        position = state.position.get(self.symbol, 0)
        
        # 如果没有订单深度数据
        if not order_depth.buy_orders and not order_depth.sell_orders:
            return orders

        logger.print(f"Current mode: {self.current_mode}")
        #普通模式
        if self.current_mode == "Normal":
            # 获取最佳买卖价格
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')

            # 计算市场公允价值
            fair_value = self.calculate_fair_value(order_depth)

            # 仓位调整系数 - 仓位越大，卖出意愿越强
            position_factor = 5 * position / self.position_limit
            adjusted_fair_value = fair_value - position_factor

            '''
            # --- Pricing factors | 价格影响因素 ------------------------------
            sunlight_bonus = 0.0
            sugar_penalty = 0.0

            if (state.observations
                    and "MAGNIFICENT_MACARONS" in state.observations.conversionObservations):
                obs = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]

                # Sunlight makes macarons more valuable | 阳光越足，马卡龙越值钱
                sunlight_bonus = max(0,
                                    obs.sunlightIndex - self.sun_threshold) * self.sun_coeff

                # High sugar price raises cost | 糖价越高，成本越高
                sugar_penalty = max(0,
                                    obs.sugarPrice - self.sugar_threshold) * self.sugar_coeff

            adjusted_fair_value = adjusted_fair_value + sunlight_bonus - sugar_penalty
            logger.print(f"delta fair value: {sunlight_bonus - sugar_penalty}")
            '''

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            # 设置买卖价格
            buy_price = best_bid + 1
            sell_price = best_ask - 1

            # 确定买入量
            available_buy = max(0, self.position_limit - position)
            if available_buy > 0:
                # 吃单逻辑：如果有比我们买价更便宜的卖单，直接吃入
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                    if ask_price < adjusted_fair_value:
                        buy_volume = min(-ask_volume, available_buy)
                        if buy_volume > 0:
                            orders.append(Order(self.symbol, ask_price, buy_volume))
                            available_buy -= buy_volume

                            if available_buy <= 0:
                                break
                    else:
                        break

                # 挂单逻辑：在当前最高买价上方挂买单
                if available_buy > 0 and buy_price < best_ask:
                    orders.append(Order(self.symbol, buy_price, available_buy))

            # 确定卖出量
            available_sell = max(0, position + self.position_limit)
            if available_sell > 0:
                # 吃单逻辑：如果有比我们卖价更高的买单，直接卖出
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > adjusted_fair_value:
                        sell_volume = min(bid_volume, available_sell)
                        if sell_volume > 0:
                            orders.append(Order(self.symbol, bid_price, -sell_volume))
                            available_sell -= sell_volume

                            if available_sell <= 0:
                                break
                    else:
                        break

                # 挂单逻辑：在当前最低卖价下方挂卖单
                if available_sell > 0 and position > 0 and sell_price > best_bid:
                    orders.append(Order(self.symbol, sell_price, -available_sell))

        #做多
        if self.current_mode == "CSI":
            if self.sunlightIndex_slope > 0 and self.sunlightIndex_history[-1] > 30: #退出条件
                self.current_mode = "Closing position"
            #做多
            max_buy_amount, _ = self.get_available_amount(self.symbol, state)
            max_buy_amount = min(max_buy_amount, self.position_limit - position)
            quick_orders, _ = self.quick_trade(self.symbol, state, max_buy_amount)
            orders.extend(quick_orders)


        #做空
        if self.current_mode == "Closing position":
            if position <= -70 and self.sunlightIndex_history[-1] > 40:
                self.current_mode = "Normal"
            else:
                _, max_sell_amount = self.get_available_amount(self.symbol, state)
                max_sell_amount = min(max_sell_amount, position + self.position_limit)
                quick_orders, _ = self.quick_trade(self.symbol, state, -max_sell_amount)
                orders.extend(quick_orders)

        return orders


    def run(self, state: TradingState) -> Tuple[List[Order], dict, int]:
        """执行策略主逻辑，同时返回订单、状态和转换请求"""
        # 生成普通订单
        orders = self.generate_orders(state)
        
        # 确定最优的转换数量
        conversions = self.determine_optimal_conversions(state)
        logger.print(f"Converting {conversions}")
        if conversions != 0:
            self.conversion_count += abs(conversions)

        # 保存策略状态
        strategy_state = self.save_state(state)

        return orders, strategy_state, conversions

    def save_state(self, state) -> dict:
        """保存策略状态"""
        return {
            "fair_value_history": list(self.fair_value_history),
            "position_history": list(self.position_history),
            "conversion_count": self.conversion_count
        }

    def load_state(self, state):
        """加载策略状态"""
        if not hasattr(state, 'traderData') or not state.traderData:
            return

        try:
            trader_data = json.loads(state.traderData)
            if self.symbol in trader_data:
                strategy_data = trader_data[self.symbol]

                if "fair_value_history" in strategy_data:
                    self.fair_value_history = deque(strategy_data["fair_value_history"], maxlen=100)

                if "position_history" in strategy_data:
                    self.position_history = deque(strategy_data["position_history"], maxlen=100)

                if "conversion_count" in strategy_data:
                    self.conversion_count = strategy_data["conversion_count"]
        except Exception as e:
            logger.print(f"Error loading state: {str(e)}")

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
            "VOLCANIC_ROCK": {
                "strategy_cls": VolcanicRockStrategy,
                "symbol": "VOLCANIC_ROCK",
                "position_limit": 400
            },
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

        trader_data.update(new_trader_data)
        trader_data = json.dumps(trader_data)
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data