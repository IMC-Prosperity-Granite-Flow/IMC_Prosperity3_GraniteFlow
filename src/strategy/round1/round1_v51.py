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


class FactorsCalculator:
    def __init__(self) -> None:
        pass

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
    def __init__(self, symbol: str, position_limit: int, ma_window: int = 120,
                 spread: float = 1.0, max_deviation: int = 200,
                 vol_threshold: float = 10, band_width: float = 25):
        self.symbol = symbol
        self.position_limit = position_limit
        self.product_config = {
            "SQUID_INK": {
                "base_quantity": 5,
                "trend_window": 120,
                "max_position": 50,
            }
        }
        self.price_history = {symbol: []}
        self.ma_short = {symbol: 0}
        self.own_order = {symbol: 0}


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
            return current_value
        else:
            return self.price_history[self.symbol][-1]


    def generate_orders(self, state: TradingState) -> List[Order]:
        """根据估计的真实价值和当前市场状况生成最佳订单"""
        orders = []

        if self.symbol not in state.order_depths:
            return orders

        # 获取当前市场数据
        order_depth = state.order_depths[self.symbol]
        current_position = state.position.get(self.symbol, 0)
        max_position = self.product_config["SQUID_INK"]["max_position"]

        # 计算当前价值并更新历史
        current_value = self.calculate_fair_value(order_depth)
        self.price_history[self.symbol].append(current_value)

        # 保持历史数据长度
        if len(self.price_history[self.symbol]) > 120:
            self.price_history[self.symbol] = self.price_history[self.symbol][-120:]

        trend_window = self.product_config["SQUID_INK"]["trend_window"]
        if len(self.price_history[self.symbol]) >= trend_window:
            window_data = self.price_history[self.symbol][-trend_window:]
            self.ma_short[self.symbol] = np.mean(window_data)
        else:
            self.ma_short[self.symbol] = current_value

        available_buy = max(0, max_position - current_position)
        available_sell = max(0, max_position + current_position)

        # 处理卖单（asks）的限价单
        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price < (self.ma_short[self.symbol] - 10):
                quantity = min(-ask_volume, available_buy)
                if quantity > 0:
                    orders.append(Order(self.symbol, ask_price, quantity))
                    available_buy -= quantity

        # 处理买单（bids）的限价单
        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > (self.ma_short[self.symbol] + 10):
                quantity = min(bid_volume, available_sell)
                if quantity > 0:
                    orders.append(Order(self.symbol, bid_price, -quantity))
                    available_sell -= quantity

        # 挂出被动做市单
        fair_value = self.ma_short[self.symbol]
        spread = 10  # 10 cents spread

        # 计算挂单价格
        buy_price = math.floor(fair_value - spread)
        sell_price = math.ceil(fair_value + spread)

        # 确保不超过仓位限制
        if available_buy > 0:
            orders.append(Order(self.symbol, buy_price, available_buy))
        if available_sell > 0:
            orders.append(Order(self.symbol, sell_price, -available_sell))

        return orders

    def save_state(self, state) -> dict:
        return {}

    def load_state(self, state):
        
        pass

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
        # 波动率计算的宽度
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
