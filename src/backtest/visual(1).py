import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
from logger import Logger
from utils import _process_data_  # 假设该函数返回 (market_data, trade_history)

from matplotlib import font_manager

# 设置字体为 SimHei（或系统中的其他中文字体）
font_path = "/Library/Fonts/SimHei.ttf"  
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def plot_orderbook_ladder(ax, bids, asks):
    """
    在给定的坐标轴 ax 上绘制订单簿快照：
      - 卖单显示在上方（红色）
      - 买单显示在下方（绿色）
    价格和挂单量将单独以黑体（加粗）显示，并设置背景框提高对比度。
    参数：
      bids: List[(price, volume)]，按价格从高到低排序（最佳买单在第一位）
      asks: List[(price, volume)]，按价格从低到高排序（最佳卖单在第一位）
    """
    ax.clear()
    rect_height = 1.0
    scale = 0.5  # 挂单量缩放比例

    num_asks = len(asks)
    num_bids = len(bids)

    # 绘制卖单：红色部分，显示在上方
    for i, (price, volume) in enumerate(asks):
        y = num_asks - 1 - i
        width = volume * scale
        ax.barh(y, width, height=rect_height, left=0,
                color='red', alpha=0.6, edgecolor='black', linewidth=1)
        # 在矩形中部单独摘出价格和volume，使用黑体加粗、白色背景提高对比度
        ax.text(width/2, y + rect_height/2, f"{price:.2f}\n({volume})",
                ha="center", va="center", color="black", fontsize=10, fontweight="bold",
                fontfamily="SimHei",
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    # 绘制买单：绿色部分，显示在下方
    for i, (price, volume) in enumerate(bids):
        y = - (i + 1)
        width = volume * scale
        ax.barh(y, width, height=rect_height, left=0,
                color='green', alpha=0.6, edgecolor='black', linewidth=1)
        ax.text(width/2, y + rect_height/2, f"{price:.2f}\n({volume})",
                ha="center", va="center", color="black", fontsize=10, fontweight="bold",
                fontfamily="SimHei",
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

    # 绘制分隔线
    ax.axhline(0, color='black', linewidth=1)

    # 设置 x 轴范围（左右对称）
    max_vol = max(
        max([v for _, v in bids], default=0),
        max([v for _, v in asks], default=0)
    )
    ax.set_xlim(0, max_vol * scale * 1.2)
    ax.set_ylim(-num_bids - 0.5, num_asks - 0.5)

    # 隐藏坐标轴刻度和边框
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def interactive_orderbook(product: str, df_orderbook: pd.DataFrame, df_trades: pd.DataFrame, logger=None):
    """
    使用 matplotlib.widgets.Slider 在 PyCharm 下实现交互式回放界面，
    显示指定产品在不同时间戳下的订单簿（上下对齐：卖单在上，买单在下）、
    交易记录和 Logger 信息。
    """
    timestamps = sorted(df_orderbook[df_orderbook['product'] == product]['timestamp'].unique())
    if not timestamps:
        print(f"没有找到产品 {product} 的数据。")
        return

    fig = plt.figure(figsize=(8, 6))
    ax_orderbook = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax_info = plt.subplot2grid((4, 1), (3, 0))
    slider_ax = plt.axes([0.15, 0.01, 0.7, 0.04])
    time_slider = Slider(ax=slider_ax, label="Time Index",
                         valmin=0, valmax=len(timestamps) - 1,
                         valinit=0, valstep=1)

    def update(val):
        time_index = int(time_slider.val)
        t = timestamps[time_index]
        row = df_orderbook[(df_orderbook['product'] == product) & (df_orderbook['timestamp'] == t)]
        if row.empty:
            ax_orderbook.clear()
            ax_orderbook.text(0.5, 0.5, f"时间 {t} 下无订单簿数据", ha='center', va='center')
            ax_info.clear()
            ax_info.text(0.5, 0.5, "无交易记录/Logger 记录", ha='center', va='center')
            fig.canvas.draw_idle()
            return
        row = row.iloc[0]
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
        bids = [(p, v) for p, v in bids if pd.notna(p) and pd.notna(v)]
        asks = [(p, v) for p, v in asks if pd.notna(p) and pd.notna(v)]
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(asks, key=lambda x: x[0])
        plot_orderbook_ladder(ax_orderbook, bids_sorted, asks_sorted)
        ax_orderbook.set_title(f"{product} Order Book at time {t}", fontsize=12)
        ax_info.clear()
        info_lines = [f"时间: {t}"]
        trades = df_trades[(df_trades["timestamp"] == t) & (df_trades["symbol"] == product)]
        if trades.empty:
            info_lines.append("无交易记录。")
        else:
            for _, trade in trades.iterrows():
                buyer = trade.get("buyer", "")
                seller = trade.get("seller", "")
                price = trade.get("price", "")
                qty = trade.get("quantity", "")
                info_lines.append(f"Buyer: {buyer}, Seller: {seller}, Price: {price}, Qty: {qty}")
        if logger is not None:
            logger_summary = logger.store()
            if product in logger_summary:
                recs = [rec for rec in logger_summary[product] if rec[0] == t]
                if recs:
                    for rec_ts, attr, value in recs:
                        info_lines.append(f"Logger - {attr}: {value}")
                else:
                    info_lines.append("无 Logger 记录。")
        info_text = "\n".join(info_lines)
        ax_info.text(0.01, 0.5, info_text, transform=ax_info.transAxes,
                     fontsize=10, verticalalignment='center')
        ax_info.axis('off')
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    update(0)
    plt.show()

def main():
    file_path = "tutorial_v1.log"
    market_data, trade_history = _process_data_(file_path)
    record = Logger()
    record.record(0, "RAINFOREST_RESIN", "Test_attr", 1024)
    interactive_orderbook("RAINFOREST_RESIN", market_data, trade_history, record)

if __name__ == "__main__":
    main()
