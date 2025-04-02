from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np

class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        print("Current position :", state.position)
        result = {}
        profit_pct_limit = 0.001
        for product in state.order_depths:
            # Now we only trade kelp, skipping rainforest_resin
            if product == 'RAINFOREST_RESIN':
                print('Skipping RAINFOREST_RESIN')
                continue
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            #acceptable_price = self.estimate_fair_price(state, product)
            #print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            best_ask, best_ask_amount = self.get_best_price(order_depth.sell_orders, "ask")
            best_bid, best_bid_amount = self.get_best_price(order_depth.buy_orders, "bid")

            print(f'best_ask: {best_ask}, best_ask_amount: {best_ask_amount}, best_bid: {best_bid}, best_bid_amount: {best_bid_amount}')

            #check position risk
            if self.check_position_risk(state.position, product):
                #market making
                fair_price, expect_bid, expect_ask = self.estimate_fair_price(state, product)
                print("Fair price : " + str(fair_price))
                print("Expect bid : " + str(expect_bid))
                print("Expect ask : " + str(expect_ask))

                if best_bid < expect_bid and best_ask > expect_ask:
                    print("Market making")
                    # 市场买卖
                    amount = min(-best_ask_amount, best_bid_amount)
                    price_gap = (best_ask - fair_price)
                    profit = price_gap * amount
                    if profit / fair_price > profit_pct_limit:
                        print("Market making profitable. Profit:", profit)
                        orders.append(Order(product, best_ask, -amount))
                        orders.append(Order(product, best_bid, amount))
                    else:
                        print("Market making not profitable. Profit:", profit)
                else:
                    print("Not in market making range")
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
    
    def estimate_fair_price(self, state: TradingState, product: str) -> int:
        # Estimate fair price based on market data and trader's observations
    
        # 用市场买卖加权均价
        order_depth = state.order_depths.get(product, OrderDepth())
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = sum(price * amount for price, amount in order_depth.buy_orders.items()) / sum(amount for price, amount in order_depth.buy_orders.items())
            best_ask = sum(price * amount for price, amount in order_depth.sell_orders.items()) / sum(amount for price, amount in order_depth.sell_orders.items())
            print('Using order depth to estimate fair price')
            fair_price = (best_bid + best_ask) / 2
            volatility = self.calculate_market_volatility(state, product)
            spread = self.calculate_spread(volatility)
            expect_bid = fair_price - spread / 2
            expect_ask = fair_price + spread / 2
            return fair_price, expect_bid, expect_ask
        else:
            print('Using default price of 2025')
            return 2025  # 如果没有数据，返回默认值



    def get_best_price(self, orders: dict, type: str):
        if not orders:
            return None, None
        if type == 'ask':
            best_price = min(orders.keys())
            best_amount = orders[best_price]
        if type == 'bid':
            best_price = max(orders.keys())
            best_amount = orders[best_price]
        return best_price, best_amount
    
    def check_position_risk(self, position:dict, product:str) -> bool:
        position_limit = 1000 #持仓限制
        if product in position and abs(position[product])  >=  position_limit:
            return False
        return True

    
    def calculate_market_volatility(self, state: TradingState, product: str) -> float:
    # 计算市场的波动性，可以用标准差来衡量
        recent_trades = state.market_trades.get(product, [])
        if len(recent_trades) > 1:
            prices = [trade.price for trade in recent_trades]
            return np.std(prices)  # 返回价格标准差作为波动性
        return 0  # 如果交易数据不足，则返回0

    def calculate_spread(self, volatility: float) -> float:
        # 根据市场波动性计算买卖差价
        if volatility > 0:
            return volatility * 2
        return 0.01  # 默认最小差价

        


    
