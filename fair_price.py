from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        print("Current position :", state.position)
        result = {}
        profit_pct_limit = 0.01
        for product in state.order_depths:
            # Now we only trade kelp, skipping rainforest_resin
            if product == 'RAINFOREST_RESIN':
                print('Skipping RAINFOREST_RESIN')
                continue
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = self.estimate_fair_price(state, product)
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

            best_ask, best_ask_amount = self.get_best_price(order_depth.sell_orders, "ask")
            best_bid, best_bid_amount = self.get_best_price(order_depth.buy_orders, "bid")

            print(f'best_ask: {best_ask}, best_ask_amount: {best_ask_amount}, best_bid: {best_bid}, best_bid_amount: {best_bid_amount}')

            #check position
            if self.check_position_risk(state.position, product):
                #ask
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if self.check_profitable(acceptable_price, best_ask, best_ask_amount, profit_pct_limit):
                        print("BUY", str(-best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_amount))
                #bid
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    if self.check_profitable(acceptable_price, best_bid, best_bid_amount, profit_pct_limit = profit_pct_limit):
                        print("SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
    
    def estimate_fair_price(self, state: TradingState, product: str) -> int:
        # Estimate fair price based on market data and trader's observations
        recent_trades = state.market_trades.get(product, [])
    
        if recent_trades:
            total_price = sum(trade.price * trade.quantity for trade in recent_trades)
            total_qty = sum(trade.quantity for trade in recent_trades)
            print('Using recent trades to estimate fair price')
            return total_price / total_qty 
        
        # 如果没有最近交易，用市场买卖加权均价
        order_depth = state.order_depths.get(product, OrderDepth())
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = sum(price * amount for price, amount in order_depth.buy_orders.items())
            best_ask = sum(price * amount for price, amount in order_depth.sell_orders.items())
            print('Using order depth to estimate fair price')
            return (best_bid + best_ask) / 2  # 取均值
    
        print('Using default price of 10')
        return 10  # 如果没有数据，返回默认值

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
        position_limit = 10
        if product in position and abs(position[product])  >=  position_limit:
            return False
        return True
    
    def check_profitable(self, acceptable_price: int, order_price: int, order_amount: int, profit_pct_limit: float):
        profit = (order_price - acceptable_price) * order_amount
        if profit > 0 and profit / acceptable_price > profit_pct_limit:
            return True
        return False

        


    
