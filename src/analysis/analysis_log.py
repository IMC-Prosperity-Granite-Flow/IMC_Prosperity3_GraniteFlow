import json
import re

log_file_path = "log/log4.log"

# 交易日志的字段
trade_fields = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]

def parse_log(log_data):
    """解析 JSON 日志"""
    parsed_data = {}

    # 解析 lambdaLog（如果存在）
    lambda_log = log_data.get("lambdaLog", log_data.get("sandboxLog", ""))
    if lambda_log:
        patterns = {
            "timestamp": re.compile(r'"timestamp": (\d+)'),
            "position": re.compile(r"Current position : (\{.*?\})"),
            "best_ask": re.compile(r"best_ask: (\d+)"),
            "best_bid": re.compile(r"best_bid: (\d+)"),
            "fair_price": re.compile(r"Fair price : ([\d.]+)"),
            "expect_bid": re.compile(r"Expect bid : ([\d.]+)"),
            "expect_ask": re.compile(r"Expect ask : ([\d.]+)"),
            "profit": re.compile(r"Profit: ([\d.]+)"),
            "market_making": re.compile(r"Market making"),
        }

        for key, pattern in patterns.items():
            match = pattern.search(lambda_log)
            parsed_data[key] = match.group(1) if match else None

        if parsed_data["position"]:
            try:
                parsed_data["position"] = json.loads(parsed_data["position"].replace("'", "\""))
            except json.JSONDecodeError:
                parsed_data["position"] = None

    # 解析交易日志（如果没有 lambdaLog）
    if not lambda_log and all(field in log_data for field in trade_fields):
        parsed_data.update({field: log_data[field] for field in trade_fields})

    return parsed_data

parsed_logs = []
with open(log_file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
    log_entries = json_pattern.findall(raw_text)

    print("提取的日志条目:", log_entries[:5])  # 打印前 5 条检查格式

    for entry in log_entries:
        entry = entry.strip()
        if entry:
            try:
                log_data = json.loads(entry)
                print("解析后的 JSON 数据:", json.dumps(log_data, indent=4, ensure_ascii=False))  # 方便调试

                parsed_log = parse_log(log_data)
                if parsed_log:
                    parsed_logs.append(parsed_log)
                else:
                    print("Warning: 该日志不符合已知格式，跳过。")

            except json.JSONDecodeError as e:
                print(f"Warning: JSON 解析失败，错误详情: {e}")
                print(f"错误的 JSON 数据: {entry}")

print(json.dumps(parsed_logs, indent=4, ensure_ascii=False))