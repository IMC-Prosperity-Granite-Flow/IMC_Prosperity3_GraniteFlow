class Logger:
    def __init__(self):
        # 内部记录结构：字典，key 为 product，value 为记录列表，每个记录是 (timestamp, attr, value)
        self.logs = {}

    def record(self, timestamp, product, attr, value):
        """记录一条日志记录"""
        if product not in self.logs:
            self.logs[product] = []
        self.logs[product].append((timestamp, attr, value))

    def store(self):
        """
        返回汇总结果：
          - 按 product 名称（升序）排序，
          - 每个 product 内部按照 timestamp 排序
        返回一个字典 { product: [(timestamp, attr, value), ...], ... }
        """
        summary = {}
        for product in sorted(self.logs.keys()):
            summary[product] = sorted(self.logs[product], key=lambda x: x[0])
        return summary
