from abc import ABC, abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict, List, Tuple, Deque
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


class FactorsCalculator:
    def __init__(self) -> None:
        pass

    def fractional_derivative(ts, alpha, n_terms=10):
        """
        计算时间序列 ts 的分数阶导数
        参数:
            ts: 时间序列
            alpha: 阶数
            n_terms: 历史项的数量，控制内存长短

        返回:
            分数阶导数序列（长度与 ts 相同，前面一些值为 nan）
        """

        def binomial_coeff(a, k):
            return math.gamma(a + 1) / (math.gamma(k + 1) * math.gamma(a - k + 1))

        ts = np.asarray(ts)
        result = np.full_like(ts, np.nan, dtype=np.float64)

        for t in range(n_terms, len(ts)):
            val = 0.0
            for k in range(n_terms):
                coeff = (-1) ** k * binomial_coeff(alpha, k)
                val += coeff * ts[t - k]
            result[t] = val
        return result

    def calculate_market_volatility(self, state: TradingState, product: str) -> float:
        """计算市场的波动率"""
        # 取最近的交易数据
        recent_trades = state.market_trades.get(product, [])
        if len(recent_trades) > 1:
            prices = [trade.price for trade in recent_trades]
            return np.std(prices)  # 返回价格标准差作为波动性
        return 0  # 如果交易数据不足，则返回0

    def calculate_volatility(self, prices) -> float:
        """计算价格序列的波动率"""
        if len(prices) > 1:
            return np.std(prices)  # 返回价格标准差作为波动性
        return 0  # 如果价格序列不足，则返回0

    def calculate_ma(self, prices: List[float], span: int):
        """
        计算移动平均线（MA)
        prices: 价格序列
        span: 移动平均线的长度
        """
        if not prices:
            return 0.0
        if span > len(prices):
            logger.print(f'Error: span {span} is larger than the length of prices {len(prices)}')
            return 0.0
        return sum(prices[-span:]) / span

    def calculate_ema(self, prices: List[float], span: int) -> float:
        """
        计算指数移动平均 (EMA)
        prices: 价格序列
        span: 移动平均线的长度
        """
        if not prices:
            return 0.0
        alpha = 2 / (span + 1)  # EMA 平滑因子
        ema = prices[0]  # 初始化为第一天的价格
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def orderbook_imbalance(self, state, product: str, fair_price: float) -> float:
        """ 计算订单簿不平衡度 """
        order_depth: OrderDepth = state.order_depths[product]

        buy_orders = [(price, amount) for price, amount in order_depth.buy_orders.items()]
        sell_orders = [(price, amount) for price, amount in order_depth.sell_orders.items()]

        # 根据价格离公平价的远近加权
        buy_pressure = sum(amount * np.exp(-(fair_price - price)) for price, amount in buy_orders if price != 0)
        sell_pressure = sum(amount * np.exp(-(price - fair_price)) for price, amount in sell_orders if price != 0)

        total_pressure = buy_pressure + sell_pressure
        if total_pressure == 0:
            return 0
        return (buy_pressure - sell_pressure) / total_pressure

    def price_momentum(self, prices: List[int], product: str) -> float:
        '''计算价格动量'''
        if len(prices) < 20:
            logger.print(f'Error: length of prices {len(prices)} is less than 20')
            return 0.0
        # 计算短期均线和长期均线
        short_ema = self.calculate_ema(prices[-20:], 5, product)
        long_ema = self.calculate_ema(prices[-20:], 20, product)
        return short_ema - long_ema


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
        logger.print("Saving State", strategy_state)

        return self.orders, strategy_state
    
    def save_state(self, state) -> dict:
        """保存策略状态"""
        return {}
    def load_state(self, data: dict):
        """加载策略状态"""
        self.trader_data = data
        pass

class KelpStrategy(Strategy):
    """海带做市策略"""

    def __init__(self, symbol: str, position_limit: int, alpha: float, beta: float, time_window: int):
        super().__init__(symbol, position_limit)
        # 添加海带策略特有参数
        self.alpha = alpha #adjusted fair price清仓系数
        self.beta = beta #adjusted fair price订单簿不平衡度系数
        self.time_window = time_window #价格序列长度
        self.position_history = deque(maxlen = self.time_window)
        self.price_history = deque(maxlen = self.time_window)
        #初始化因子计算器
        self.calculator = FactorsCalculator()
        self.trader_data = {}
    

    def calculate_moving_average(self, span: int):
        """计算移动平均线"""
        ma = self.calculator.calculate_ma(self.price_history, span)
        return ma
    
    def calculate_mid_price(self, state: TradingState):
        """计算中间价"""
        order_depth = state.order_depths[self.symbol]
        buy_prices = list(order_depth.buy_orders.keys())
        sell_prices = list(order_depth.sell_orders.keys())
        best_ask = min(sell_prices) if sell_prices else 0
        best_bid = max(buy_prices) if buy_prices else 0
        logger.print(f"Best ask {best_ask}, Best bid {best_bid}")
        mid_price = (best_ask + best_bid) / 2
        logger.print(f"Mid price {mid_price}")
        return mid_price
            
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
        take_position1 = 0
        take_position2 = 0
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]
        fair_value = self.calculate_fair_value(order_depth)
        orderbook_imbalance = self.calculator.orderbook_imbalance(state, self.symbol, fair_value)
        #调整公允价格
        fair_value = fair_value + self.alpha * position + self.beta * orderbook_imbalance

        available_buy = max(0, self.position_limit - position)
        available_sell = max(0, self.position_limit + position)

        orders = []

        # 吃单逻辑
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < fair_value:
                # 计算最大可买量
                buyable = min(-vol, self.position_limit - position)
                if buyable > 0:
                    orders.append(Order(self.symbol, ask, buyable))
                    take_position1 += buyable
            else:
                break  # 后续价格更高，不再处理

        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > fair_value:
                # 计算最大可卖量
                sellable = min(vol, self.position_limit + position)
                if sellable > 0:
                    orders.append(Order(self.symbol, bid, -sellable))
                    take_position2 += sellable
            else:
                break  # 后续价格更低，不再处理

        # 挂单逻辑
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        desired_bid = best_bid + 1
        if desired_bid >= fair_value:
            desired_bid = round(fair_value) - 1

        desired_ask = best_ask - 1
        if desired_ask <= fair_value:
            desired_ask = round(fair_value) + 1

        desired_buy = available_buy - take_position1
        desired_sell = available_sell - take_position2

        if desired_buy > 0:
            orders.append(Order(self.symbol, desired_bid, desired_buy))
        if desired_sell > 0:
            orders.append(Order(self.symbol, desired_ask, -desired_sell))
        logger.print(
            f"Current position: {position}, take_position1: {take_position1}, take_position2: {take_position2}")
        return orders
    
    def save_state(self, state) -> dict:
        return_dict = {}
        position = state.position.get(self.symbol)
        if self.position_history:
            return_dict['position'] = self.position_history
        else:
            return_dict['position'] = []
        return_dict['position'].append(position)

        #保存mid_price
        mid_price = self.calculate_mid_price(state)
        logger.print(f"Calculated mid price: {mid_price}")
        self.price_history.append(mid_price)
        #维护长度
        if len(self.price_history) > self.time_window: 
                self.price_history.popleft()
        logger.print(f"Saving, Price history: {self.price_history}")
        return return_dict
    
    def load_state(self, data: dict):
        #data为历史数据，类型为字典
        self.trader_data = data
        self.position_history = self.trader_data.get('position', {})
        logger.print(f"Loading, Price history: {self.price_history}")
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

    def load_state(self, data: dict):
        pass

class SquidInkStrategy(Strategy):
    """SQUIDINK策略"""
    def __init__(self, symbol: str, position_limit: int, reversal_threshold: int, trend_window: int, value_window: int,
                 cycle_length: int, base_spread: int, min_spread: int, position_scaling: float, price_momentum_factor: float):
        super().__init__(symbol, position_limit)

        #策略参数
        self.reversal_threshold = reversal_threshold
        self.trend_window = trend_window
        self.value_window = value_window
        self.cycle_length = cycle_length
        self.base_spread = base_spread
        self.min_spread = min_spread
        self.position_scaling = position_scaling
        self.price_momentum_factor = price_momentum_factor

        #历史数据
        self.price_history = []
        self.price_predictions = []
        self.ma_short = []
        self.ma_long = []
        self.last_fair_value = []
        self.current_phase = []  # 1表示上升趋势，-1表示下降趋势
        self.phase_changes = []  # 跟踪相位变化
        self.last_crossover = []  # 均线最后一次交叉的时间点
        self.cycle_position = []  # 在价格周期中的位置
    
    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        """基于订单簿和历史数据计算估计的真实价值"""
        
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

        # 更新历史数据
        
        self.price_history = []
        self.last_fair_value = current_value
        self.current_phase = 0
        self.phase_changes = []
        self.cycle_position  = 0
        
        # 将当前值添加到历史记录中
        self.price_history.append(current_value)
        
        # 只保留最近的历史数据
        history_limit = max(self.trend_window * 2, self.value_window)
        if len(self.price_history) > history_limit:
            self.price_history = self.price_history[-history_limit:]
        
        # 计算短期和长期移动平均线
        if len(self.price_history) >= self.trend_window:
            self.ma_short = np.mean(self.price_history[-self.trend_window:])
        else:
            self.ma_short = current_value
            
        if len(self.price_history) >= self.value_window:
            self.ma_long = np.mean(self.price_history[-self.value_window:])
        else:
            self.ma_long = current_value
        
        # 检测趋势阶段
        prev_phase = self.current_phase
        if self.ma_short > self.ma_long:
            self.current_phase = 1  # 上升趋势
        elif self.ma_short < self.ma_long:
            self.current_phase = -1  # 下降趋势
        
        # 跟踪相位变化
        if prev_phase != self.current_phase and prev_phase != 0:
            self.phase_changes.append(len(self.price_history))
            
            # 在相位变化时重置周期位置
            self.cycle_position = 0
        else:
            self.cycle_position += 1
        
        # 计算加权真实价值，整合趋势和周期信息
        trend_factor = 1.0
        if len(self.phase_changes) >= 2:
            # 根据典型周期长度进行调整
            avg_cycle = np.mean(np.diff(self.phase_changes))
            cycle_progress = self.cycle_position / self.cycle_length
            
            # 当接近典型周期长度时预测反转
            if self.current_phase == 1 and cycle_progress > 0.7:
                trend_factor = max(0.8, 1.5 - cycle_progress)
            elif self.current_phase == -1 and cycle_progress > 0.7:
                trend_factor = min(1.2, 0.5 + cycle_progress)
        
        # 结合短期和长期移动平均线与近期动量
        momentum = 0
        if len(self.price_history) >= 3:
            # 最近价格变动方向和强度
            recent_change = self.price_history[-1] - self.price_history[-3]
            momentum = recent_change * self.price_momentum_factor
        
        # 计算最终真实价值
        fair_value = (
            self.ma_short * 0.4 + 
            self.ma_long * 0.6 + 
            momentum
        ) * trend_factor
        
        # 保存以供下次迭代使用
        self.last_fair_value = fair_value
        
        return fair_value

    def generate_orders(self, state: TradingState) -> List[Order]:
        """根据估计的真实价值和当前市场状况生成最佳订单"""
        order_depth = state.order_depths[self.symbol]
        fair_value = self.calculate_fair_value(order_depth)
        position = state.position.get(self.symbol, 0)
        orders = []
        
        # 查找当前最佳买入价/卖出价
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        
        # 计算市场中间价和价差
        midpoint = (best_bid + best_ask) / 2 if best_bid and best_ask else fair_value
        spread = best_ask - best_bid if best_bid and best_ask else self.min_spread * 2
        
        # 根据真实价值和价差确定期望的买入/卖出价
        desired_spread = max(self.min_spread, self.base_spread + abs(position) * self.position_scaling)
        
        # 基于持仓的调整（逆向持仓倾向）
        position_adjustment = -position * self.position_scaling
        
        # 根据真实价值、价差和持仓调整买入/卖出价
        desired_bid = int(fair_value + position_adjustment - desired_spread / 2)
        desired_ask = int(fair_value + position_adjustment + desired_spread / 2)
        
        # 确保我们的买入价有竞争力但不过高
        if desired_bid >= best_bid and desired_bid < fair_value:
            bid_price = best_bid + 1
        else:
            bid_price = desired_bid
            
        # 确保我们的卖出价有竞争力但不过低
        if desired_ask <= best_ask and desired_ask > fair_value:
            ask_price = best_ask - 1
        else:
            ask_price = desired_ask
        
        # 确定持仓限制和可用容量
        position_limit = self.position_limit
        available_buy = max(0, position_limit - position)
        available_sell = max(0, position_limit + position)
        
        # 机会主义交易 - 积极吃单获取有利价格
        for ask_price, volume in sorted(order_depth.sell_orders.items()):
            # 如果卖价明显低于真实价值，则买入
            if ask_price < fair_value - self.min_spread:
                buy_volume = min(abs(volume), available_buy)
                if buy_volume > 0:
                    orders.append(Order(self.symbol, ask_price, buy_volume))
                    available_buy -= buy_volume
        
        for bid_price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            # 如果买价明显高于真实价值，则卖出
            if bid_price > fair_value + self.min_spread:
                sell_volume = min(volume, available_sell)
                if sell_volume > 0:
                    orders.append(Order(self.symbol, bid_price, -sell_volume))
                    available_sell -= sell_volume
        
        # 做市交易 - 在价差附近挂限价单
        if available_buy > 0:
            orders.append(Order(self.symbol, bid_price, available_buy))
            
        if available_sell > 0:
            orders.append(Order(self.symbol, ask_price, -available_sell))
        
        return orders

    def save_state(self, state):
        return {}
    
    def load_state(self, data):
        pass

class Config:
    def __init__(self):
        self.PRODUCT_CONFIG = {
        "KELP": {
            "strategy_cls": KelpStrategy,
            "position_limit": 50,
            "alpha": -0.03,
            "beta": 0,
            "time_window": 20
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
            "reversal_threshold": 20,    # 考虑价格反转信号的阈值
            "trend_window": 10,          # 趋势计算的窗口大小
            "value_window": 50,          # 计算真实价值的窗口大小
            "cycle_length": 200,         # 预期的价格周期长度
            "base_spread": 2,            # 基础价差
            "min_spread": 5,             # 最小可接受的价差
            "position_scaling": 0.8,     # 基于持仓的调整因子
            "price_momentum_factor": 0.1 # 价格动量调整因子
        }
    }
    

class Trader:
    # config
    '''
    PRODUCT_CONFIG = {
        "KELP": {
            "strategy_cls": KelpStrategy,
            "position_limit": 50,
            "alpha": -0.03,
            "beta": 0,
            "time_window": 20
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
            "reversal_threshold": 20,    # 考虑价格反转信号的阈值
            "trend_window": 10,          # 趋势计算的窗口大小
            "value_window": 50,          # 计算真实价值的窗口大小
            "cycle_length": 200,         # 预期的价格周期长度
            "base_spread": 2,            # 基础价差
            "min_spread": 5,             # 最小可接受的价差
            "position_scaling": 0.8,     # 基于持仓的调整因子
            "price_momentum_factor": 0.1 # 价格动量调整因子
        }
    }
    '''
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
                strategy.load_state(trader_data[product])
            if product in state.order_depths:
                product_orders, strategy_state = strategy.run(state)
                orders[product] = product_orders
                
                new_trader_data[product] = strategy_state

        trader_data.update(new_trader_data)
        trader_data = json.dumps(trader_data)
        logger.print("Current trader_data", trader_data)
        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data
