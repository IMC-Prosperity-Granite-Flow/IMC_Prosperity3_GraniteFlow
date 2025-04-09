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

    def calculate_volatility(self, prices: list, span) -> float:
        """计算价格序列的波动率"""
        prices = prices[-span:]
        if len(prices) > 1:
            return np.std(prices)  # 返回价格标准差作为波动性
        return 0  # 如果价格序列不足，则返回0

    def calculate_ma(self, prices: List[float], span: int):
        """
        计算移动平均线（MA)
        prices: 价格序列
        span: 移动平均线的长度
        """
        if type(prices) != list:
            prices = list(prices)
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
        if type(prices) != list:
            prices = list(prices)
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
            return 0
        # 计算短期均线和长期均线
        short_ema = self.calculate_ema(prices[-20:], 5, product)
        long_ema = self.calculate_ema(prices[-20:], 20, product)
        return short_ema - long_ema
    
    def get_mid_reversion_gap(self, mid_prices: list, mid_prices_20: list):
        '''计算mid price回归距离'''
        return mid_prices[-1] - mid_prices_20[-1]


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
        mid_price = (best_ask + best_bid) / 2

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
        self.price_history.append(mid_price)
        #维护长度
        if len(self.price_history) > self.time_window: 
                self.price_history.popleft()
        return return_dict
    
    def load_state(self, state):
        pass

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
    """SQUIDINK策略"""
    def __init__(self, symbol: str, position_limit: int, time_window: int, alpha: float, deviation_threshold: float, predict_length: int):
        super().__init__(symbol, position_limit)

        self.time_window = time_window
        self.alpha = -0.03
        self.deviation_threshold = deviation_threshold
        self.predict_length = predict_length

        self.fair_value_history = Deque(maxlen=time_window)
        self.mid_price_history = Deque(maxlen=time_window)
        self.fair_value_ma10 = Deque(maxlen=time_window)
        self.fair_value_ma20 = Deque(maxlen=time_window)
        self.mid_price_history_ma20 = Deque(maxlen=time_window)
        self.mid_reversion_gap = Deque(maxlen=time_window)
        self.log_return5 = Deque(maxlen=time_window)
        self.calculator = FactorsCalculator()

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        bid1_price = max(order_depth.buy_orders.keys())
        ask1_price = min(order_depth.sell_orders.keys())
        return (bid1_price + ask1_price) / 2

    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        """基于订单簿前三档的加权中间价计算"""

        sell_orders = [(price, amount) for price, amount in order_depth.sell_orders.items() if amount != 0]
        buy_orders = [(price, amount) for price, amount in order_depth.buy_orders.items() if amount != 0]
        #计算加权均价
        weighted_price = sum(price * amount for price, amount in buy_orders) + sum(price * -amount for price, amount in sell_orders)
        sum_amount = sum(amount for price, amount in buy_orders) + sum(-amount for price, amount in sell_orders)
        fair_value = weighted_price / sum_amount if sum_amount > 0 else 0
        #保存历史数据
        return fair_value
    
    def generate_orders(self, state):
        orders = []
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        buy_orders = [(price, amount) for price, amount in order_depth.buy_orders.items() if amount != 0]
        sell_orders = [(price, amount) for price, amount in order_depth.sell_orders.items() if amount != 0]

        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid] 
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = order_depth.sell_orders[best_ask]
        spread = best_ask - best_bid

        vol = self.calculator.calculate_volatility(list(self.fair_value_history), 5)

        fair_value = self.calculate_fair_value(order_depth)
        fair_value_ma10 = self.calculator.calculate_ma(self.fair_value_history, 10) if len(self.fair_value_history) > 10 else fair_value
        

        #清仓机制
        adjusted_fair_value = fair_value + self.alpha * position



        logger.print("Taking")
        if len(self.log_return5) == self.time_window and all (x != 0 for x in list(self.log_return5)):
            #如果连续预测收益为正，买入
            if all(x > 0 for x in list(self.log_return5)[-self.predict_length:]):
                max_return_long = 0.052
                logger.print("All log returns are positive, taking position")
                mean_return = np.mean(list(self.log_return5)[-self.predict_length:])
                max_buy_amount = max(0, self.position_limit - position)
                return_ratio = mean_return / max_return_long
                if return_ratio > 0.05:
                    buy_amount = max_buy_amount
                    if buy_amount > 0:
                        logger.print(f"Buying at {best_ask - 1}, amount: {buy_amount}")
                        orders.append(Order(self.symbol, best_ask - 1, buy_amount))
                    position += buy_amount
            #如果连续预测收益为负数，卖出
            if all(x < 0 for x in list(self.log_return5)[-self.predict_length:]):
                max_return_short = -0.055
                logger.print("All log returns are negative, taking position")
                mean_return = np.mean(list(self.log_return5)[-self.predict_length:])
                return_ratio = mean_return / max_return_short
                if return_ratio > 0.05:
                    max_sell_amount = max(0, position + self.position_limit)
                    sell_amount = max_sell_amount
                    if sell_amount > 0:
                        logger.print(f"Selling at {best_bid + 1}, amount: {sell_amount}")
                        orders.append(Order(self.symbol, best_bid + 1, -sell_amount))
                        position -= sell_amount
        
            


        '''
        logger.print(f"Position: {position}")
        if position > - self.position_limit:
            logger.print(f"Position limit: {self.position_limit}")
            available_sell = max(0, self.position_limit + position)
            logger.print(f"Available sell: {available_sell}")
            orders.append(Order(self.symbol, best_bid + 1, -available_sell))

        '''

        '''
        fair_value_deviation = fair_value - fair_value_ma10

        if fair_value_deviation > self.deviation_threshold:
            logger.print(f"Fair value deviation: {fair_value_deviation} larger than threshold: {self.deviation_threshold}")
            available_sell = max(0, position + self.position_limit)
            if available_sell > 0:
                sell_amount = min(available_sell, best_bid_amount)
                logger.print(f"Available sell: {available_sell}, sell at {best_bid + 1}, amount: {sell_amount}")
                orders.append(Order(self.symbol, best_bid + 1, -sell_amount))
        
        if fair_value_deviation < - self.deviation_threshold:
            logger.print(f"Fair value deviation: {fair_value_deviation} larger than threshold: {self.deviation_threshold}")
            available_buy = max(0, self.position_limit - position)
            if available_buy > 0:
                buy_amount = min(available_buy, best_ask_amount)
                logger.print(f"Available buy: {available_buy}, buy at {best_ask - 1}, amount: {buy_amount}")
                orders.append(Order(self.symbol, best_ask - 1, buy_amount))
        '''
        '''
        #吃单
        if best_ask < adjusted_fair_value:
            logger.print(f"Best ask: {best_ask} smaller than adjusted_fair_value: {adjusted_fair_value}")
            # 计算最大可买量
            buyable = min(-order_depth.sell_orders[best_ask], self.position_limit - position)
            if buyable > 0:
                print(f"Buyable, {buyable}, buy at {best_ask + 1}")
                orders.append(Order(self.symbol, best_ask + 1, buyable))
                position += buyable
        
        if best_bid > adjusted_fair_value:
            logger.print(f"Best bid: {best_bid} larger than adjusted_fair_value: {adjusted_fair_value}")
            # 计算最大可卖量
            sellable = min(order_depth.buy_orders[best_bid], self.position_limit + position)
            if sellable > 0:
                logger.print(f"Sellable, {sellable}, sell at {best_bid - 1}")
                orders.append(Order(self.symbol, best_bid - 1, -sellable))
                position -= sellable
        '''
        
        '''
        #计算偏离值
        fair_value_deviation = fair_value - fair_value_ma10

        #均值回归策略
        if fair_value_deviation > self.deviation_threshold:
            logger.print(f"Fair value deviation: {fair_value_deviation} larger than threshold: {self.deviation_threshold}")
            # 卖出
            desired_sell = max(0, position + self.position_limit)
            if desired_sell > 0:
                logger.print(f"Desired sell, {desired_sell}, sell at {best_bid - 1}")
                orders.append(Order(self.symbol, best_bid - 1, -desired_sell))
                position -= desired_sell
        elif fair_value_deviation < -self.deviation_threshold:
            logger.print(f"Fair value deviation: {fair_value_deviation} smaller than threshold: {self.deviation_threshold}")
            # 买入
            desired_buy = max(0, self.position_limit - position)
            if desired_buy > 0:
                logger.print(f"Desired buy, {desired_buy}, buy at {best_ask + 1}")
                orders.append(Order(self.symbol, best_ask + 1, desired_buy))
                position += desired_buy
        
        #做市
        '''

        return orders

    def save_state(self, state):
        return {}
    
    
    def load_state(self, state):

        #保存fair_value
        self.fair_value_history.append(self.calculate_fair_value(state.order_depths[self.symbol]))
        if len(self.fair_value_history) > self.time_window:
            self.fair_value_history.popleft()
        #logger.print("fair value history: ", self.fair_value_history)
        #保存ma10
        self.fair_value_ma10.append(self.calculator.calculate_ma(list(self.fair_value_history), 10))
        if len(self.fair_value_ma10) > self.time_window:
            self.fair_value_ma10.popleft()
        #logger.print("fair value ma10: ", self.fair_value_ma10)
        #保存ma20
        self.fair_value_ma20.append(self.calculator.calculate_ma(list(self.fair_value_history), 20))
        if len(self.fair_value_ma20) > self.time_window:
            self.fair_value_ma20.popleft()
        #logger.print("fair value ma20: ", self.fair_value_ma20)

        #保存mid_price
        mid_price = self.calculate_mid_price(state.order_depths[self.symbol])
        self.mid_price_history.append(mid_price)
        if len(self.mid_price_history) > self.time_window:
            self.mid_price_history.popleft()
        #logger.print("mid price history: ", self.mid_price_history)
        #保存mid_price_ma20
        mid_price_history_ma20 = self.calculator.calculate_ma(list(self.mid_price_history), 20)
        self.mid_price_history_ma20.append(mid_price_history_ma20)
        if len(self.mid_price_history_ma20) > self.time_window:
            self.mid_price_history_ma20.popleft()
        #logger.print("mid price history ma20: ", self.mid_price_history_ma20)
        #保存mid_reversion_gap
        mid_reversion_gap = self.calculator.get_mid_reversion_gap(self.mid_price_history, self.mid_price_history_ma20)
        self.mid_reversion_gap.append(mid_reversion_gap)
        if len(self.mid_reversion_gap) > self.time_window:
            self.mid_reversion_gap.popleft()
        #logger.print("mid reversion gap: ", self.mid_reversion_gap)

        #保存预测的log_return5
        a = 3.777e-6
        b = 0.0003
        if all(x != 0 for x in self.mid_price_history and self.mid_price_history_ma20):
            log_return5 = a + b * mid_reversion_gap
        else:
            log_return5 = 0
        self.log_return5.append(log_return5)
        if len(self.log_return5) > self.time_window:
            self.log_return5.popleft()
        logger.print("log return5: ", self.log_return5)

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
            "time_window": 20,             # 计算均价的时长
            "alpha": -0.03,                # 公允价格偏移
            "deviation_threshold": 15,   # 偏离均值回归的阈值
            "predict_length": 5           #预测时使用历史数据窗口长度
        }
    }
    

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
