import pandas as pd
import tushare as ts
from typing import List, Dict
from datetime import datetime, time

class HighConfidenceReturnStrategy:
    def is_market_closed(self):
        """
        判断市场是否已经收盘。
        这里只是示例实现，实际应用中需要根据具体的数据源和交易时间规则进行调整。
        """

        # 假设交易时间是9:30到15:00
        market_open = time(9, 30)
        market_close = time(15, 0)
        now = datetime.now().time()

        return not (market_open <= now <= market_close)
    def __init__(self, target_return: float):
        self.target_return = target_return

    

    def get_real_time_sell2_price(self, token, stock_code):
        # 设置 tushare token
        ts.set_token(token)
        pro = ts.pro_api()
        try:
            # 获取实时行情数据
            df = pro.realtime_quote(ts_code=stock_code)
            if not df.empty and 'ask2' in df.columns:
                return float(df['ask2'].iloc[0])
            else:
                print('未获取到实时卖2价格')
                return 0
        except Exception as e:
            print(f'获取实时卖2价格时出错: {e}')
            return 0

    def calculate_signals(self, df: pd.DataFrame) -> List[Dict]:
        # 获取实时卖2价或开盘价
        if self.is_market_closed() and 'open' in df.columns:
            current_price = df['open'].iloc[1]
        else:
            # 假设从配置文件或其他地方获取 token 和 stock_code，这里需要根据实际情况修改
            token = 'your_tushare_token'
            stock_code = df['code'].iloc[0]
            current_price = self.get_real_time_sell2_price(token, stock_code)
        """
        计算当天是否有99.99%置信度获得超过目标的收益，并给出买卖价。
        df: 包含股票历史数据的DataFrame
        """
        # 示例：假设已经计算出置信区间
        # 计算历史价格的99.99%置信区间
        # lower_bound = df['close'].quantile(0.0001)
        # upper_bound = df['close'].quantile(0.9999)
        
        # lower_bound和upper_bound替换成均值置信度99.99%的上下界
        import numpy as np
        from scipy import stats
        mean = df['close'].mean()
        std = df['close'].std()
        z_score = stats.norm.ppf((1 + 0.9999) / 2)
        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std

        # 计算是否有99.99%置信度实现3%收益
        potential_return = (upper_bound - current_price) / current_price

        valid_trades = []


        if (upper_bound - current_price) / current_price >= self.target_return:
            buy_price = current_price
            sell_price = current_price * (1 + self.target_return)
            valid_trades.append({
                "stock_code": df['code'].iloc[0],
                "stock_name": df['name'].iloc[0],
                "buy_price": buy_price,
                "sell_price": "upper:"+str(upper_bound)+"  "+"建议:"+str(buy_price*(1+self.target_return)),
                "model": "高置信度收益策略",
                "actual_return": f'{potential_return:.2%}'
            })

        return valid_trades
    