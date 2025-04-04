import io
import json

import  pandas as pd
def _process_data_(file):
    with open(file, 'r') as file:
        log_content = file.read()
    sections = log_content.split('Sandbox logs:')[1].split('Activities log:')
    sandbox_log =  sections[0].strip()
    activities_log = sections[1].split('Trade History:')[0]
    # sandbox_log_list = [json.loads(line) for line in sandbox_log.split('\n')]
    trade_history =  json.loads(sections[1].split('Trade History:')[1])
    # sandbox_log_df = pd.DataFrame(sandbox_log_list)
    market_data_df = pd.read_csv(io.StringIO(activities_log), sep=";", header=0)
    trade_history_df = pd.json_normalize(trade_history)
    return market_data_df, trade_history_df

def extract_activities_log(file_path: str) -> pd.DataFrame:
    """
    从日志文件中提取 Activities log 部分并转换为 DataFrame。
    日志文件格式示例：
        Sandbox logs:
        ...（Sandbox log 内容）

        Activities log:
        <CSV 格式数据，分隔符为";">

        Trade History:
        ...（Trade log 内容）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 定位 Activities log 部分和 Trade History 部分的起始位置
    act_idx = content.find("Activities log:")
    trade_idx = content.find("Trade History:")

    if act_idx == -1:
        print("未找到 'Activities log:' 部分")
        return pd.DataFrame()

    start = act_idx + len("Activities log:")
    if trade_idx == -1:
        csv_text = content[start:]
    else:
        csv_text = content[start:trade_idx]

    csv_text = csv_text.strip()
    if not csv_text:
        print("提取到的 Activities log 文本为空")
        return pd.DataFrame()

    csv_io = io.StringIO(csv_text)
    try:
        df = pd.read_csv(csv_io, sep=";")
    except Exception as e:
        print("解析 CSV 时发生错误:", e)
        return pd.DataFrame()
    return df


def load_market_data_from_log(log_file_path: str) -> pd.DataFrame:
    """
    利用 extract_activities.py 中的函数提取活动日志部分，并返回 DataFrame。
    该 DataFrame 应包含至少: [timestamp, symbol, bid_price_1, bid_volume_1, ask_price_1, ask_volume_1]
    """
    df_activities = extract_activities_log(log_file_path)
    # 如果活动日志里还有更多档位 (bid_price_2, bid_volume_2 等)，可以自行扩展
    return df_activities


def extract_submission_trade_history_log(file_path: str) -> pd.DataFrame:
    """
    从日志文件中提取 Trade History 部分的 CSV 数据，
    并只返回 buyer 或 seller 字段为 "SUBMISSION" 的记录。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    trade_idx = content.find("Trade History:")
    if trade_idx == -1:
        print("未找到 'Trade History:' 部分")
        return pd.DataFrame()
    # 取 Trade History 部分文本（从 'Trade History:' 到文件末尾）
    trade_text = content[trade_idx + len("Trade History:"):].strip()
    if not trade_text:
        print("提取到的 'Trade History' 文本为空")
        return pd.DataFrame()
    trade_io = io.StringIO(trade_text)
    try:
        df_trade = pd.read_csv(trade_io, sep=";")
    except Exception as e:
        print("解析 Trade History CSV 时发生错误:", e)
        return pd.DataFrame()
    # 过滤出 buyer 或 seller 为 "SUBMISSION" 的记录
    df_submission = df_trade[(df_trade["buyer"] == "SUBMISSION") | (df_trade["seller"] == "SUBMISSION")]
    return df_submission


