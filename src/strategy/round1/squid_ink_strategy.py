from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict
import numpy as np
import json
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

        # 我们将state.traderData、trader_data和self.logs截断到相同的最大长度，以适应日志限制
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


class Trader:
    def __init__(self):
        self.product_config = {
            "SQUID_INK": {
                "max_position": 20,          # 最大持仓量
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
        self.price_history = {}
        self.price_predictions = {}
        self.ma_short = {}
        self.ma_long = {}
        self.last_true_value = {}
        self.current_phase = {}  # 1表示上升趋势，-1表示下降趋势
        self.phase_changes = {}  # 跟踪相位变化
        self.last_crossover = {}  # 均线最后一次交叉的时间点
        self.cycle_position = {}  # 在价格周期中的位置

    def calculate_true_value(self, symbol: str, order_depth: OrderDepth) -> float:
        """基于订单簿和历史数据计算估计的真实价值"""
        config = self.product_config[symbol]
        
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
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.last_true_value[symbol] = current_value
            self.current_phase[symbol] = 0
            self.phase_changes[symbol] = []
            self.cycle_position[symbol] = 0
        
        # 将当前值添加到历史记录中
        self.price_history[symbol].append(current_value)
        
        # 只保留最近的历史数据
        history_limit = max(config["trend_window"] * 2, config["value_window"])
        if len(self.price_history[symbol]) > history_limit:
            self.price_history[symbol] = self.price_history[symbol][-history_limit:]
        
        # 计算短期和长期移动平均线
        if len(self.price_history[symbol]) >= config["trend_window"]:
            self.ma_short[symbol] = np.mean(self.price_history[symbol][-config["trend_window"]:])
        else:
            self.ma_short[symbol] = current_value
            
        if len(self.price_history[symbol]) >= config["value_window"]:
            self.ma_long[symbol] = np.mean(self.price_history[symbol][-config["value_window"]:])
        else:
            self.ma_long[symbol] = current_value
        
        # 检测趋势阶段
        prev_phase = self.current_phase[symbol]
        if self.ma_short[symbol] > self.ma_long[symbol]:
            self.current_phase[symbol] = 1  # 上升趋势
        elif self.ma_short[symbol] < self.ma_long[symbol]:
            self.current_phase[symbol] = -1  # 下降趋势
        
        # 跟踪相位变化
        if prev_phase != self.current_phase[symbol] and prev_phase != 0:
            self.phase_changes[symbol].append(len(self.price_history[symbol]))
            
            # 在相位变化时重置周期位置
            self.cycle_position[symbol] = 0
        else:
            self.cycle_position[symbol] += 1
        
        # 计算加权真实价值，整合趋势和周期信息
        trend_factor = 1.0
        if len(self.phase_changes[symbol]) >= 2:
            # 根据典型周期长度进行调整
            avg_cycle = np.mean(np.diff(self.phase_changes[symbol]))
            cycle_progress = self.cycle_position[symbol] / config["cycle_length"]
            
            # 当接近典型周期长度时预测反转
            if self.current_phase[symbol] == 1 and cycle_progress > 0.7:
                trend_factor = max(0.8, 1.5 - cycle_progress)
            elif self.current_phase[symbol] == -1 and cycle_progress > 0.7:
                trend_factor = min(1.2, 0.5 + cycle_progress)
        
        # 结合短期和长期移动平均线与近期动量
        momentum = 0
        if len(self.price_history[symbol]) >= 3:
            # 最近价格变动方向和强度
            recent_change = self.price_history[symbol][-1] - self.price_history[symbol][-3]
            momentum = recent_change * config["price_momentum_factor"]
        
        # 计算最终真实价值
        true_value = (
            self.ma_short[symbol] * 0.4 + 
            self.ma_long[symbol] * 0.6 + 
            momentum
        ) * trend_factor
        
        # 保存以供下次迭代使用
        self.last_true_value[symbol] = true_value
        
        return true_value

    def get_best_orders(self, symbol: str, true_value: float, order_depth: OrderDepth, position: int) -> List[Order]:
        """根据估计的真实价值和当前市场状况生成最佳订单"""
        config = self.product_config[symbol]
        orders = []
        
        # 查找当前最佳买入价/卖出价
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
        
        # 计算市场中间价和价差
        midpoint = (best_bid + best_ask) / 2 if best_bid and best_ask else true_value
        spread = best_ask - best_bid if best_bid and best_ask else config["min_spread"] * 2
        
        # 根据真实价值和价差确定期望的买入/卖出价
        desired_spread = max(config["min_spread"], config["base_spread"] + abs(position) * config["position_scaling"])
        
        # 基于持仓的调整（逆向持仓倾向）
        position_adjustment = -position * config["position_scaling"]
        
        # 根据真实价值、价差和持仓调整买入/卖出价
        desired_bid = int(true_value + position_adjustment - desired_spread / 2)
        desired_ask = int(true_value + position_adjustment + desired_spread / 2)
        
        # 确保我们的买入价有竞争力但不过高
        if desired_bid >= best_bid and desired_bid < true_value:
            bid_price = best_bid + 1
        else:
            bid_price = desired_bid
            
        # 确保我们的卖出价有竞争力但不过低
        if desired_ask <= best_ask and desired_ask > true_value:
            ask_price = best_ask - 1
        else:
            ask_price = desired_ask
        
        # 确定持仓限制和可用容量
        max_position = config["max_position"]
        available_buy = max(0, max_position - position)
        available_sell = max(0, max_position + position)
        
        # 机会主义交易 - 积极吃单获取有利价格
        for ask_price, volume in sorted(order_depth.sell_orders.items()):
            # 如果卖价明显低于真实价值，则买入
            if ask_price < true_value - config["min_spread"]:
                buy_volume = min(abs(volume), available_buy)
                if buy_volume > 0:
                    orders.append(Order(symbol, ask_price, buy_volume))
                    available_buy -= buy_volume
        
        for bid_price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            # 如果买价明显高于真实价值，则卖出
            if bid_price > true_value + config["min_spread"]:
                sell_volume = min(volume, available_sell)
                if sell_volume > 0:
                    orders.append(Order(symbol, bid_price, -sell_volume))
                    available_sell -= sell_volume
        
        # 做市交易 - 在价差附近挂限价单
        if available_buy > 0:
            orders.append(Order(symbol, bid_price, available_buy))
            
        if available_sell > 0:
            orders.append(Order(symbol, ask_price, -available_sell))
        
        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """游戏调用的主方法"""
        result = {}
        conversions = 0
        trader_data = json.dumps({
            "price_history": {k: v[-5:] if len(v) > 5 else v for k, v in self.price_history.items()},
            "current_phase": self.current_phase,
            "cycle_position": self.cycle_position
        })
        
        symbol = "SQUID_INK"
        
        # 检查是否有此交易对的市场数据
        if symbol in state.order_depths:
            order_depth = state.order_depths[symbol]
            position = state.position.get(symbol, 0)
            
            # 只有当同时存在买单和卖单时才交易
            if order_depth.buy_orders and order_depth.sell_orders:
                # 基于订单簿和历史数据计算真实价值
                true_value = self.calculate_true_value(symbol, order_depth)
                
                # 根据真实价值和当前市场生成订单
                orders = self.get_best_orders(symbol, true_value, order_depth, position)
                
                # 将订单添加到结果中
                result[symbol] = orders
                
                # 记录一些信息
                logger.print(f"SQUID_INK - 持仓: {position}, 真实价值: {true_value:.2f}")
                logger.print(f"阶段: {self.current_phase.get(symbol, 0)}, 周期位置: {self.cycle_position.get(symbol, 0)}")
                if symbol in self.ma_short and symbol in self.ma_long:
                    logger.print(f"短期均线: {self.ma_short[symbol]:.2f}, 长期均线: {self.ma_long[symbol]:.2f}")
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data 