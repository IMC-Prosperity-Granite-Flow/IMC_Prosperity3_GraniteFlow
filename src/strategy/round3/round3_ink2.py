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



class SquidInkStrategy(Strategy):
    def __init__(self, symbol: str, position_limit: int, ma_window: int, threshold: float):
        super().__init__(symbol, position_limit)
        self.ma_window = ma_window
        self.price_history = deque(maxlen = ma_window)
        self.threshold = threshold
        self.current_mode = "No_action"

    def calculate_fair_value(self, order_depth):
        """计算mid_price"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 2000
        #取出所有不是nan的price
        buy_prices = [price for price in order_depth.buy_orders if price is not None]
        sell_prices = [price for price in order_depth.sell_orders if price is not None]
        mid_price = (sum(buy_prices) + sum(sell_prices)) / (len(buy_prices) + len(sell_prices))
        return mid_price
    
    def generate_orders(self, state):
        logger.print("current mode", self.current_mode)
        orders = []
        fair_value = self.calculate_fair_value(state.order_depths[self.symbol])
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        ma = np.mean(self.price_history)
        diff = fair_value - ma
        
        if self.current_mode == "Needle" and abs(diff) < self.threshold * 0.5:
            #平仓
            if position > 0:
                for price, amount in order_depth.buy_orders.items():
                    amount = min(amount, position)
                    orders.append(Order(self.symbol, price, -amount))
                    position -= amount
            if position < 0:
                for price, amount in order_depth.sell_orders.items():
                    amount = min(-amount, -position)
                    orders.append(Order(self.symbol, price, -amount))
                    position += (-amount)
            if position == 0:
                self.current_mode = "No_action"

        if len(self.price_history) < self.ma_window or abs(diff) <= self.threshold or self.current_mode == "No_action":
            #做市
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            best_ask_amount = order_depth.sell_orders[best_ask]
            mid_price = (best_bid + best_ask) / 2
            spread_threshold = 2
            buy_price = round(mid_price - spread_threshold/2)
            sell_price = round(mid_price + spread_threshold/2)
            if best_ask - best_bid <= 3:
                return orders
            else:
                safe_bid_amount = min(best_bid_amount, self.position_limit - position)
                safe_ask_amount = min(-best_ask_amount, self.position_limit + position)
                if safe_bid_amount > 0:
                    orders.append(Order(self.symbol, buy_price, safe_bid_amount))
                if safe_ask_amount > 0:
                    orders.append(Order(self.symbol, sell_price, -safe_ask_amount))

        elif abs(diff) > self.threshold:
            #检测插针
            self.current_mode = "needle"
            if diff > 0:
                #卖出
                for price, amount in order_depth.buy_orders.items():
                    orders.append(Order(self.symbol, price, -amount))
            else:
                #买入
                for price, amount in order_depth.sell_orders.items():
                    orders.append(Order(self.symbol, price, -amount))

    
            



        
        return orders
        #trend
        '''
        ma = np.mean(self.price_history)
        diff = ma - fair_value

        #平仓策略
        if abs(diff) < self.threshold * 0.3 and position != 0:
            #平仓
            if position > 0:
                for price, amount in order_depth.buy_orders.items():
                    amount = min(amount, position)
                    orders.append(Order(self.symbol, price, -amount))
                    position -= amount

            if position < 0:
                for price, amount in order_depth.sell_orders.items():
                    amount = min(-amount, -position)
                    orders.append(Order(self.symbol, price, amount))
                    position += amount

        #开仓策略


        if diff > self.threshold:
            #买入
            for price, amount in order_depth.sell_orders.items():
                amount = -amount
                amount = int(min(amount, self.position_limit - position) * min((diff - self.threshold) / (2 * self.threshold), 1))
                if amount > 0:
                    orders.append(Order(self.symbol, price, amount))
                    position += amount 
        elif diff < -self.threshold:
            #卖出
            for price, amount in order_depth.buy_orders.items():
                amount = int(min(amount, position + self.position_limit) * min((-diff - self.threshold) / (2 * self.threshold), 1))
                if amount > 0:
                    orders.append(Order(self.symbol, price, -amount))
                    position -= amount

        

        return orders

        '''
                




    def load_state(self, state):
        #维护price history
        fair_value = self.calculate_fair_value(state.order_depths[self.symbol])
        self.price_history.append(fair_value)

'''
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
        self.take_spread = 10

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
        
        eat_pos1 = 0
        eat_pos2 = 0

        fair_value = self.calculate_fair_value(state.order_depths[self.symbol])
        # 保存fair_value
        self.fair_value_history.append(fair_value)
        if len(self.fair_value_history) > self.ma_window:
            self.fair_value_history.popleft()

        if len(self.fair_value_history) >= 200:
            ma= np.mean(list(self.fair_value_history)[-200:])
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
                    if bid_price >= ma - 10 :
                            quantity = min(bid_volume, current_position)
                            orders.append(Order(self.symbol, bid_price, -quantity))
                            current_position -= quantity
                            if current_position == 0: break
                return orders

            elif fair_value <= ma + 10 and current_position < 0 and self.needle_direction == 1:
                for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                     if ask_price <= ma - 10 :
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
                            if quantity > 0 and current_position + quantity < 45: # 仓位控制
                                orders.append(Order(self.symbol, ask_price, quantity))
                                available_buy -= quantity
                        if ask_price < ma - 120: # 极端情况，少量仓位拉低均价
                            quantity = min(-ask_volume, available_buy)
                            if quantity > 0:
                                orders.append(Order(self.symbol, ask_price, quantity))
                                available_buy -= quantity


                elif self.needle_direction == 1:  # 上涨插针
                    for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                        if bid_price > ma + 30:
                            quantity = min(bid_volume, available_sell)
                            if quantity > 0 and current_position + quantity > -45: # 仓位控制
                                orders.append(Order(self.symbol, bid_price, -quantity))
                                available_sell -= quantity
                        if bid_price > ma + 120: # 极端情况，少量仓位抬高均价
                            quantity = min(bid_volume, available_sell)
                            if quantity > 0:
                                orders.append(Order(self.symbol, bid_price, -quantity))
                                available_sell -= quantity
            return orders
        # 波幅超标检测
        else:
            # 下跌插针检测
            if best_ask < ma - 30: # 已检测到波动
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
            elif best_bid > ma + 30: # 已检测到波动
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > ma + 30:  # 高点
                        quantity = min(bid_volume, available_sell)
                        if quantity > 0:
                            orders.append(Order(self.symbol, bid_price, -quantity))
                            available_sell -= quantity
                self.current_mode = "action"
                self.needle_direction = 1
                return orders
            
            else:
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
                    if ask_price < (ma - self.take_spread):
                        quantity = min(-ask_volume, available_buy)
                        if quantity > 0:
                            orders.append(Order(self.symbol, ask_price, quantity))
                            available_buy -= quantity
                            eat_pos1 += quantity
                            #logger.print(f"buy {quantity} at {ask_price}")

                # 处理买单（bids）的限价单
                for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price > (ma + self.take_spread):
                        quantity = min(bid_volume, available_sell)
                        if quantity > 0:
                            orders.append(Order(self.symbol, bid_price, -quantity))
                            available_sell -= quantity
                            eat_pos2 += quantity
                            #logger.print(f"sell {quantity} at {bid_price}")

                # 计算挂单价格
                buy_price = math.floor(ma - self.take_spread)
                sell_price = math.ceil(ma + self.take_spread)

                if current_position + eat_pos1 > 0 and best_bid >ma:
                    quantity = min(best_bid_amount, round(0.7*available_sell))
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_bid, -quantity))
                        available_sell -= quantity

                if current_position - eat_pos2< 0 and best_ask <ma:
                    quantity = min(-best_ask_amount, round(0.7*available_buy))
                    if quantity > 0:
                        orders.append(Order(self.symbol, best_ask, quantity))
                        available_sell -= quantity

                if available_buy > 0:
                    orders.append(Order(self.symbol, buy_price, available_buy))
                if available_sell > 0:
                    orders.append(Order(self.symbol, sell_price, -available_sell))

                return orders
        return orders

    def save_state(self, state):
        return {}

    def load_state(self, state):
        pass
'''


class Config:
    def __init__(self):
        self.PRODUCT_CONFIG = {
            "SQUID_INK": {
                "strategy_cls": SquidInkStrategy,
                "symbol": "SQUID_INK",
                "position_limit": 50,  # 最大持仓量
                "ma_window": 200,  # 计算均价的时长
                "threshold": 40
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
