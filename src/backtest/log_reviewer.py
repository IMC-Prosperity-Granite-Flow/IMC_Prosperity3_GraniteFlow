import io
import json
from logger import  Logger
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd

from utils import _process_data_,extract_activities_log

def print_orderbook_and_trades(df_orderbook: pd.DataFrame,
                               df_trades: pd.DataFrame,
                               only_with_trade: bool = False,
                               logger: Logger = None):
    """
    逐行打印每个时间戳、产品的订单簿信息和对应的交易记录，并显示该产品在当前时间戳下的 Logger 记录。

    参数说明：
      - df_orderbook: 包含订单簿信息的 DataFrame，列如：
            day, timestamp, product,
            bid_price_1, bid_volume_1, bid_price_2, bid_volume_2, bid_price_3, bid_volume_3,
            ask_price_1, ask_volume_1, ask_price_2, ask_volume_2, ask_price_3, ask_volume_3
      - df_trades: 包含交易记录的 DataFrame，列如：
            timestamp, symbol, price, quantity, buyer, seller, ...
      - only_with_trade: 若为 True，则只打印存在交易记录的订单簿快照
      - submission_filter: 过滤条件，只显示 buyer 或 seller 为 submission_filter 的交易记录
      - logger: Logger 对象，用于记录并显示额外的属性信息。
                打印时只显示当前产品、当前时间戳匹配的记录。
    """
    # 按 day, timestamp, product 排序
    df_orderbook_sorted = df_orderbook.sort_values(by=["day", "timestamp", "product"])
    # 获取 Logger 汇总结果
    logger_summary = logger.store() if logger is not None else {}

    for idx, row in df_orderbook_sorted.iterrows():
        day = row["day"]
        ts = row["timestamp"]
        product = row["product"]

        # 筛选交易记录：匹配 timestamp 和 product (假设 trade_history 中 symbol 对应 product)
        trades_this_ts = df_trades[
            (df_trades["timestamp"] == ts) & (df_trades["symbol"] == product)
            ]


        # 若 only_with_trade 为 True 且当前没有交易记录，则跳过
        if only_with_trade and trades_this_ts.empty:
            continue

        # ============ 打印订单簿 ============
        # 假设有 3 档买单和卖单
        bids = [
            (row["bid_price_1"], row["bid_volume_1"]),
            (row["bid_price_2"], row["bid_volume_2"]),
            (row["bid_price_3"], row["bid_volume_3"]),
        ]
        asks = [
            (row["ask_price_1"], row["ask_volume_1"]),
            (row["ask_price_2"], row["ask_volume_2"]),
            (row["ask_price_3"], row["ask_volume_3"]),
        ]
        # 买单按价格从高到低排序；卖单按价格从低到高排序
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(asks, key=lambda x: x[0])

        print("=" * 70)
        print(f"Day: {day}, Timestamp: {ts}, Product: {product}")
        print("--- BIDS (买单) [从高到低] ---")
        for price, volume in bids_sorted:
            print(f"   Price: {price}, Volume: {volume}")
        print("--- ASKS (卖单) [从低到高] ---")
        for price, volume in asks_sorted:
            print(f"   Price: {price}, Volume: {volume}")

        # ============ 打印交易记录 ============
        if trades_this_ts.empty:
            print("\n-- Trades / Orders --")
            print("   No trades (with submission) at this timestamp/product.")
        else:
            print("\n-- Trades / Orders --")
            for t_idx, trade_row in trades_this_ts.iterrows():
                buyer = trade_row.get("buyer", "")
                seller = trade_row.get("seller", "")
                price = trade_row.get("price", None)
                qty = trade_row.get("quantity", None)
                print(f"   Buyer: {buyer}, Seller: {seller}, Price: {price}, Qty: {qty}")

        # ============ 打印 Logger 汇总结果 ============
        # 只显示 logger 中记录 timestamp 与当前订单簿记录匹配的结果
        if logger is not None and product in logger_summary:
            recs = [rec for rec in logger_summary[product] if rec[0] == ts]
            if recs:
                print("\n-- Logger Summary (for this product at timestamp {}) --".format(ts))
                for rec in recs:
                    rec_ts, attr, value = rec
                    print(f"   {attr}: {value}")
            else:
                print("\n-- Logger Summary: No logger records for this timestamp. --")

        print("=" * 70)
        print()  # 空行分隔


if __name__ == "__main__":
    # 替换成实际的日志文件路径
    file_path = "src/backtest/tutorial_v1.log"
    market_data, trade_history = _process_data_(file_path)
    record = Logger()

    #record.record(199600,"RAINFOREST_RESIN","Test_attr",1024)



    if not market_data.empty:
        print_orderbook_and_trades(market_data,trade_history,True,record)
