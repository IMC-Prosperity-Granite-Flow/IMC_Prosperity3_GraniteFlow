from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

# 4951
class Trader:
    # 参数配置分离（每个品种独立配置）
    PRODUCT_CONFIG = {
        "RAINFOREST_RESIN": {
            "base_offset": 3,    # 基础报价偏移
            "max_position": 50,  # 最大持仓
            "eat_position1":0,    # 固定的吃单的仓位（初始值）
            "eat_position2": 0,
            "level2spread":8,    # spread超过这个值就用另一个offset
        },
        "KELP": {
            "max_position": 50,  # 最大持仓
            "eat_position":0,    # 固定的吃单的仓位（初始值）
            "eat_vol": 30        # 如果有吃单增加的仓位
        }
    }

    def __init__(self):
        # 策略路由字典（产品名: 对应的策略方法）
        self.strategy_router = {
            "RAINFOREST_RESIN": self.rainforestresin_strategy,
            "KELP": self.kelp_strategy
        }

    def calculate_true_value(self, order_depth: OrderDepth) -> float:
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

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result = {}
        conversions = 10
        trader_data = "SAMPLE"

        for product in ["RAINFOREST_RESIN", "KELP"]:

            # 策略调用
            od = state.order_depths.get(product, None)
            if not od or not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue


            if product in self.strategy_router:
                strategy = self.strategy_router[product]
                result[product] = strategy(
                    od,
                    state.position.get(product, 0),
                    product,
                )
        # logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def rainforestresin_strategy(self, order_depth: OrderDepth, current_pos: int, product: str) -> List[Order]:
        config = self.PRODUCT_CONFIG[product]
        orders = []
        max_pos = config["max_position"]
        offset = config["base_offset"]
        eat_pos1 = config["eat_position1"]
        eat_pos2 = config["eat_position2"]
        level2spread = config["level2spread"]
        eat_vol = config["eat_vol"]
        FIXED_MID = 10000  # 固定中间价
        # 吃单逻辑 ================================================
        # 处理所有低于10000的卖单（按价格升序排列）
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < FIXED_MID:
                # 计算最大可买量
                buyable = min(-vol, max_pos - current_pos)
                if buyable > 0:
                    orders.append(Order(product, ask, buyable))
                    eat_pos1 += buyable
            else:
                break  # 后续价格更高，不再处理

        # 处理所有高于10000的买单（按价格降序排列）
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > FIXED_MID:
                # 计算最大可卖量
                sellable = min(vol, max_pos + current_pos)
                if sellable > 0:
                    orders.append(Order(product, bid, -sellable))
                    eat_pos2 += sellable
            else:
                break  # 后续价格更低，不再处理

        # 挂单逻辑 ================================================
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread = best_ask - best_bid
        if spread > level2spread:
            offset = 4
        desired_bid = FIXED_MID - offset  # 固定买单价
        desired_ask = FIXED_MID + offset  # 固定卖单价

        # 计算可用挂单量
        available_buy = max(0, max_pos - current_pos)
        available_sell = max(0, max_pos + current_pos)
        desired_buy = available_buy - eat_pos1
        desired_sell = available_sell - eat_pos2  # 固定吃单额度
        # 买盘挂单（正数表示买入）
        if desired_buy > 0:
            orders.append(Order(product, desired_bid, desired_buy))

        # 卖盘挂单（负数表示卖出）
        if desired_sell > 0:
            orders.append(Order(product, desired_ask, -desired_sell))

        return orders

    def kelp_strategy(self, order_depth: OrderDepth, current_pos: int, product: str) -> List[Order]:
        config = self.PRODUCT_CONFIG[product]
        orders = []
        max_pos = config["max_position"]
        eat_pos = config["eat_position"]
        eat_vol = config["eat_vol"]

        true_value = self.calculate_true_value(order_depth)
        # 吃单逻辑
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < true_value:
                # 计算最大可买量
                buyable = min(-vol, max_pos - current_pos)
                if buyable > 0:
                    orders.append(Order(product, ask, buyable))
                    eat_pos += eat_vol
            else:
                break  # 后续价格更高，不再处理

        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > true_value:
                # 计算最大可卖量
                sellable = max(-vol, -max_pos - current_pos)
                if sellable < 0:
                    orders.append(Order(product, bid, sellable))
                    eat_pos += eat_vol
            else:
                break  # 后续价格更低，不再处理

        # 挂单逻辑
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price= (best_bid+best_ask)/2
        spread = best_ask - best_bid
        if spread > 2:
            desired_bid = best_bid + 1
            desired_ask = best_ask - 1

            available_buy = max(0, max_pos - current_pos)
            available_sell = max(0, max_pos + current_pos)
            desired_buy = available_buy - eat_pos
            desired_sell = available_sell - eat_pos

            if desired_buy > 0:
                orders.append(Order(product, desired_bid, desired_buy))
            if desired_sell > 0:
                orders.append(Order(product, desired_ask, -desired_sell))
        return orders
