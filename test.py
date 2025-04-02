from datamodel import OrderDepth, TradingState, Order, Trade
from imc_prosperity3.fair_price import Trader


# 模拟数据

from datamodel import OrderDepth, TradingState, Order, Trade
from imc_prosperity3.fair_price import Trader

# 模拟数据
order_depths = {
    "PRODUCT1": OrderDepth(),  # 先实例化一个 OrderDepth 对象
    "PRODUCT2": OrderDepth(),
}

# 手动为 OrderDepth 对象设置 buy_orders 和 sell_orders
order_depths["PRODUCT1"].buy_orders = {10: 5, 9: 3}
order_depths["PRODUCT1"].sell_orders = {11: -4, 12: -8}

order_depths["PRODUCT2"].buy_orders = {140: 3, 141: 5}
order_depths["PRODUCT2"].sell_orders = {143: -6, 144: -2}

'''order_depths = {
    "PRODUCT1": OrderDepth(
        buy_orders={10: 5, 9: 3},
        sell_orders={11: -4, 12: -8}
    ),
    "PRODUCT2": OrderDepth(
        buy_orders={140: 3, 141: 5},
        sell_orders={143: -6, 144: -2}
    ),
}'''

market_trades = {
    "PRODUCT1": [Trade(symbol="PRODUCT1", price=10, quantity=3, buyer="", seller="", timestamp=1000)],
    "PRODUCT2": [],
}

position = {
    "PRODUCT1": 5,
    "PRODUCT2": -3
}

trading_state = TradingState(
    traderData="Test Data",
    timestamp=1000,
    listings={},
    order_depths=order_depths,
    own_trades={},
    market_trades=market_trades,
    position=position,
    observations={}
)


trader = Trader()

result, conversions, traderData = trader.run(trading_state)

# 打印结果
print("Result:", result)
print("Conversions:", conversions)
print("Trader Data:", traderData)