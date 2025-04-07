from abc import ABC, abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict, List, Tuple, TypeAlias
import numpy as np
import json
import jsonpickle
import math
from collections import deque

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


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


class IndicatorsCalculater:
    def __init__(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: json,
                 product) -> None:
        self.state = state
        self.orders = orders
        self.conversions = conversions
        self.trader_data = trader_data
        self.product = product

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


from abc import ABC, abstractmethod
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Deque, Tuple
import json
from collections import deque


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

    def __init__(self, symbol: str, position_limit: int, alpha: float, beta):
        super().__init__(symbol, position_limit)
        # 添加海带策略特有参数
        self.alpha = alpha #adjusted fair price清仓系数
        self.alpha = beta #adjusted fair price订单簿不平衡度系数
        self.trader_data = {}
        self.position_history = []

            
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
        logger.print(
            f"Current position: {state.position.get(self.symbol, 0)}, take_position1: {take_position1}, take_position2: {take_position2}")
        current_position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]
        fair_value = self.calculate_fair_value(order_depth)

        #调整公允价格
        adjusted_fair_value = fair_value + self.alpha * current_position

        available_buy = max(0, self.position_limit - current_position)
        available_sell = max(0, self.position_limit + current_position)

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
            f"Current position: {current_position}, take_position1: {take_position1}, take_position2: {take_position2}")
        return orders
    
    def save_state(self, state) -> dict:
        return_dict = {}
        position = state.position.get(self.symbol)
        if self.position_history:
            return_dict['position'] = self.position_history
        else:
            return_dict['position'] = []
        return_dict['position'].append(position)
        return return_dict
    
    def load_state(self, data: dict):
        #data为历史数据，类型为字典
        self.trader_data = data
        self.position_history = self.trader_data.get('position', {})
        return self.position_history

class RainforestResinStrategy(Strategy):
    """树脂动态做市策略"""

    def __init__(self, symbol: str, position_limit: int,
                 make_width: float = 3.5, take_width: float = 1,
                 time_window: int = 10):
        super().__init__(symbol, position_limit)
        # 策略参数
        self.make_width = make_width
        self.take_width = take_width
        self.time_window = time_window

        # 状态存储
        self.price_history: Deque[float] = deque(maxlen=time_window)
        self.vwap_history: Deque[dict] = deque(maxlen=time_window)

    def calculate_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if len([price for price in order_depth.sell_orders.keys() if
                    abs(order_depth.sell_orders[price]) >= min_vol]) == 0 or len(
                    [price for price in order_depth.buy_orders.keys() if
                     abs(order_depth.buy_orders[price]) >= min_vol]) == 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if
                                abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max(
                    [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
            return mid_price

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int,
                             product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float,
                             width: int) -> List[Order]:
        """
        清仓逻辑
        参数:
            orders: 订单列表
            order_depth: 订单薄
            position: 当前持仓
            position_limit: 持仓限制
            product: 标的物
            buy_order_volume: 买单总量
            sell_order_volume: 卖单总量
            fair_value: 公允价格
            width: 清仓宽度
        返回：
            buy_order_volume: 买单总量
            sell_order_volume: 卖单总量
        """
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        # fair_for_ask = fair_for_bid = fair

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                # clear_quantity = position_after_take
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                # clear_quantity = abs(position_after_take)
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def generate_orders(self, state: TradingState) -> List[Order]:

        logger.print("Generating orders for Rainforest Resin")
        orders: List[Order] = []
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        logger.print(f"Position: {position}")
        logger.print(f"Order depth: {order_depth}")

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            logger.print("Calculating fair value using mid price")
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if
                            abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if
                            abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid

            mmmid_price = (mm_ask + mm_bid) / 2
            self.price_history.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[
                best_bid]) / volume
            self.vwap_history.append({"vol": volume, "vwap": vwap})

            if len(self.vwap_history) > self.time_window:
                self.vwap_history.pop(0)

            if len(self.price_history) > self.time_window:
                self.price_history.pop(0)

            fair_value = sum([x["vwap"] * x['vol'] for x in self.vwap_history]) / sum(
                [x['vol'] for x in self.vwap_history])

            fair_value = mmmid_price

            # take all orders we can
            # for ask in order_depth.sell_orders.keys():
            #     if ask <= fair_value - take_width:
            #         ask_amount = -1 * order_depth.sell_orders[ask]
            #         if ask_amount <= 20:
            #             quantity = min(ask_amount, position_limit - position)
            #             if quantity > 0:
            #                 orders.append(Order("RAINFOREST_RESIN", ask, quantity))
            #                 buy_order_volume += quantity

            # for bid in order_depth.buy_orders.keys():
            #     if bid >= fair_value + take_width:
            #         bid_amount = order_depth.buy_orders[bid]
            #         if bid_amount <= 20:
            #             quantity = min(bid_amount, position_limit + position)
            #             if quantity > 0:
            #                 orders.append(Order("RAINFOREST_RESIN", bid, -1 * quantity))
            #                 sell_order_volume += quantity

            # only taking best bid/ask

            if best_ask <= fair_value - self.take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, self.position_limit - position)
                    if quantity > 0:
                        orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + self.take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, self.position_limit + position)
                    if quantity > 0:
                        orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position,
                                                                            self.position_limit, "RAINFOREST_RESIN",
                                                                            buy_order_volume, sell_order_volume,
                                                                            fair_value, 2)

            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2

            buy_quantity = self.position_limit - (position + buy_order_volume)

            if buy_quantity > 0:
                logger.print(f"Buying {buy_quantity} contracts at {bbbf + 1}")
                orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))  # Buy order
            else:
                logger.print(f"No need to buy contracts at {bbbf + 1}")
            sell_quantity = self.position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                logger.print(f"Selling {sell_quantity} contracts at {baaf - 1}")
                orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))  # Sell order
            else:
                logger.print(f"No need to sell contracts at {baaf - 1}")
            logger.print(f"Resin Orders: {orders}")
        return orders

    def _update_market_data(self, order_depth: OrderDepth, fair_value: float):
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # 计算VWAP
        total_volume = (-order_depth.sell_orders[best_ask] +
                        order_depth.buy_orders[best_bid])
        vwap = (best_bid * (-order_depth.sell_orders[best_ask]) +
                best_ask * order_depth.buy_orders[best_bid]) / total_volume

        # 修正字段名称
        self.price_history.append((best_ask + best_bid) / 2)
        self.vwap_history.append({
            "timestamp": len(self.vwap_history),
            "vwap": vwap,
            "volume": total_volume
        })
    
    def save_state(self, state) -> dict:
        return {
            # 使用统一命名字段
            "price_series": list(self.price_history),
            "vwap_series": list(self.vwap_history),
            "strategy_params": {
                "make_width": self.make_width,
                "take_width": self.take_width
            }
        }

    def load_state(self, data: dict):
        if data:
            self.price_history = deque(data.get("price_series", []),
                                       maxlen=self.time_window)
            self.vwap_history = deque(data.get("vwap_series", []),
                                      maxlen=self.time_window)
            params = data.get("strategy_params", {})
            self.make_width = params.get("make_width", 3.5)
            self.take_width = params.get("take_width", 1)

class SquidInkStrategy(Strategy):
    """SQUIDINK策略"""
    def __init__(self, symbol: str, position_limit: int, timewindow: int = 10):
        super().__init__(symbol, position_limit)

        self.timewindow = timewindow
        #历史数据（如果需要使用的话）
        self.history = Deque[float] = deque(maxlen=timewindow)
    
    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        return 0
    
    def generate_orders(self, state):
        return {}

    def save_state(self, state):
        return {}
    
    def load_state(self, data):
        pass
    
    

class Trader:
    # config
    PRODUCT_CONFIG = {
        "KELP": {
            "strategy_cls": KelpStrategy,
            "position_limit": 50,
            "alpha": 0
            "beta": 0
        },
        "RAINFOREST_RESIN": {
            "strategy_cls": RainforestResinStrategy,
            "position_limit": 50,
            "make_width": 3.5,
            "take_width": 1,
            "time_window": 10
        }
    }

    def __init__(self):
        self.strategies = {}
        self._init_strategies()

    def _init_strategies(self):
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
