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
    在给定的坐标轴 ax 上，绘制“梯形堆叠”形状的订单簿快照。
    bids: List[(price, volume)], 从高到低排序
    asks: List[(price, volume)], 从低到高排序
    """
    ax.clear()

    rect_height = 1.0

    # 画买单 (绿色)，最优买单在顶部
    for i, (price, volume) in enumerate(bids):
        width = volume * 0.5  # 可根据数据量调整
        # x 从 -width 到 0，y 从 i*rect_height 到 (i+1)*rect_height
        ax.barh(y=i, width=width, left=-width, height=rect_height,
                color='green', alpha=0.6, edgecolor='black', linewidth=1)
        # 在矩形中部标注 价格与量
        cx = -width / 2
        cy = i + rect_height / 2
        ax.text(cx, cy, f"{price:.2f}\n({volume})",
                ha="center", va="center", color="white", fontsize=9)

    # 画卖单 (红色)，最优卖单在顶部
    for i, (price, volume) in enumerate(asks):
        width = volume * 0.5
        # x 从 0 到 width，y 从 i*rect_height 到 (i+1)*rect_height
        ax.barh(y=i, width=width, left=0, height=rect_height,
                color='red', alpha=0.6, edgecolor='black', linewidth=1)
        cx = width / 2
        cy = i + rect_height / 2
        ax.text(cx, cy, f"{price:.2f}\n({volume})",
                ha="center", va="center", color="white", fontsize=9)

    # 翻转 y 轴，让最优价在最上方
    ax.invert_yaxis()

    # 设定 x, y 范围，让图看起来更紧凑
    max_vol = max(
        max([v for _, v in bids], default=0),
        max([v for _, v in asks], default=0)
    )
    ax.set_xlim(-max_vol*0.6, max_vol*0.6)

    # y 的最大档位
    max_len = max(len(bids), len(asks))
    ax.set_ylim(-0.5, max_len + 0.5)

    # 隐藏坐标轴刻度/边框
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def interactive_orderbook(product: str, df_orderbook: pd.DataFrame, df_trades: pd.DataFrame, logger=None):
    """
    使用 matplotlib.widgets.Slider 在 PyCharm 下实现交互式回放界面，
    显示指定产品在不同时间戳下的订单簿(梯形堆叠形状)、交易记录和 Logger 信息。
    """
    # 获取该产品所有的时间戳
    timestamps = sorted(df_orderbook[df_orderbook['product'] == product]['timestamp'].unique())
    if not timestamps:
        print(f"没有找到产品 {product} 的数据。")
        return

    # 创建一个包含两个主要区域的 figure：
    # - 上方 ax_orderbook：显示梯形订单簿
    # - 下方 ax_info：显示交易记录与 Logger 信息（文本）
    fig = plt.figure(figsize=(8, 6))
    ax_orderbook = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax_info = plt.subplot2grid((4, 1), (3, 0))

    # 在底部添加 Slider
    slider_ax = plt.axes([0.15, 0.01, 0.7, 0.04])
    time_slider = Slider(ax=slider_ax, label="Time Index",
                         valmin=0, valmax=len(timestamps) - 1,
                         valinit=0, valstep=1)

    def update(val):
        time_index = int(time_slider.val)
        t = timestamps[time_index]

        # 从 df_orderbook 中提取当前快照
        row = df_orderbook[(df_orderbook['product'] == product) & (df_orderbook['timestamp'] == t)]
        if row.empty:
            ax_orderbook.clear()
            ax_orderbook.text(0.5, 0.5, f"时间 {t} 下无订单簿数据", ha='center', va='center')
            ax_info.clear()
            ax_info.text(0.5, 0.5, "无交易记录/Logger 记录", ha='center', va='center')
            fig.canvas.draw_idle()
            return

        row = row.iloc[0]

        # 提取买单与卖单数据（假设有3档）
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
        # 去除 NaN
        bids = [(p, v) for p, v in bids if pd.notna(p) and pd.notna(v)]
        asks = [(p, v) for p, v in asks if pd.notna(p) and pd.notna(v)]
        # 排序
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(asks, key=lambda x: x[0])

        # 绘制梯形订单簿
        plot_orderbook_ladder(ax_orderbook, bids_sorted, asks_sorted)
        ax_orderbook.set_title(f"{product} Order Book at time {t}", fontsize=12)

        # 交易记录 & Logger 信息
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

    # Slider 回调
    time_slider.on_changed(update)
    # 初始绘制
    update(0)
    plt.show()

def main():
    file_path = "tutorial_v1.log"
    # 从日志文件中解析数据，返回 market_data (订单簿数据) 和 trade_history (交易记录)
    market_data, trade_history = _process_data_(file_path)

    # 创建 Logger 并记录一个示例
    record = Logger()
    record.record(0, "RAINFOREST_RESIN", "Test_attr", 1024)

    # 启动交互式回放界面（在 PyCharm 中显示带滑条的窗口），
    # 绘制梯形堆叠形状的订单簿可视化
    interactive_orderbook("RAINFOREST_RESIN", market_data, trade_history, record)

    # 如果还想在终端打印只含有交易的快照，可以自行调用：
    # print_orderbook_and_trades(market_data, trade_history, only_with_trade=True, logger=record)

if __name__ == "__main__":
    main()
