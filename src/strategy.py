import pandas as pd
import tushare as ts
import yaml
import numpy as np
from scipy import stats
from pathlib import Path
from typing import List, Dict, Callable, Type
from datetime import datetime, time
from abc import ABC, abstractmethod

# 置信区间计算策略的抽象基类
class ConfidenceIntervalStrategy(ABC):
    @abstractmethod
    def calculate_bounds(self, df: pd.DataFrame) -> tuple:
        """计算置信区间上下界"""
        pass

# 给予MACD策略的置信区间计算策略
class YACDConfidenceIntervalStrategy(ConfidenceIntervalStrategy):
    def __init__(self, confidence_level: float = 0.9999):
        self.confidence_level = confidence_level

    def calculate_bounds(self, df: pd.DataFrame) -> tuple:
        """使用给予MACD策略的置信区间计算方法"""
        # 假设给予MACD策略的置信区间计算方法为：
        # 计算MACD的标准差
        df['macd_std'] = df['macd'].rolling(window=20).std()
        # 计算置信区间
        df['lower_bound'] = df['macd'] - df['macd_std'] * stats.norm.ppf((1 + self.confidence_level) / 2)
        df['upper_bound'] = df['macd'] + df['macd_std'] * stats.norm.ppf((1 + self.confidence_level) / 2)
        # 返回置信区间上下界
        return df['lower_bound'].iloc[-1], df['upper_bound'].iloc[-1]
        


# 基于正态分布的置信区间计算策略
class NormalDistributionStrategy(ConfidenceIntervalStrategy):
    def __init__(self, confidence_level: float = 0.9999):
        self.confidence_level = confidence_level
        
    def calculate_bounds(self, df: pd.DataFrame) -> tuple:
        """使用正态分布计算置信区间"""
        mean = df['close'].mean()
        std = df['close'].std()
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std
        return lower_bound, upper_bound

# 基于分位数的置信区间计算策略
class QuantileStrategy(ConfidenceIntervalStrategy):
    def __init__(self, lower_quantile: float = 0.0001, upper_quantile: float = 0.9999):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
    def calculate_bounds(self, df: pd.DataFrame) -> tuple:
        """使用分位数计算置信区间"""
        lower_bound = df['close'].quantile(self.lower_quantile)
        upper_bound = df['close'].quantile(self.upper_quantile)
        return lower_bound, upper_bound

# 策略工厂，用于创建置信区间计算策略
class StrategyFactory:
    _strategies = {
        'normal': NormalDistributionStrategy,
        'quantile': QuantileStrategy,
        'yacd': YACDConfidenceIntervalStrategy
    }
    
    @classmethod
    def get_strategy(cls, strategy_name: str, **kwargs) -> ConfidenceIntervalStrategy:
        """根据策略名称创建相应的策略实例"""
        if strategy_name not in cls._strategies:
            raise ValueError(f"不支持的策略: {strategy_name}，可用策略: {list(cls._strategies.keys())}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(**kwargs)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[ConfidenceIntervalStrategy]):
        """注册新的策略类"""
        cls._strategies[name] = strategy_class

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
    def __init__(self, target_return: float = None, confidence_strategy: str = None, strategy_params: dict = None):
        """初始化策略
        
        Args:
            target_return: 目标收益率，如果为None则从goal.yaml加载
            confidence_strategy: 置信区间计算策略名称，如果为None则从strategy_config.yaml加载
            strategy_params: 置信区间计算策略参数，如果为None则从strategy_config.yaml加载
        """
        # 从goal.yaml加载目标收益率
        if target_return is None:
            goal_path = Path(r"c:\Users\jimjiang\PycharmProjects\AlphaDog\goal.yaml")
            if not goal_path.exists():
                raise FileNotFoundError(f"未找到goal.yaml，请检查路径：{goal_path}")
            
            with open(goal_path, "r", encoding="utf-8") as f:
                goal_config = yaml.safe_load(f)
            
            # 解析百分比字符串为浮点数
            target_return_str = goal_config.get("target_return", "3%")
            self.target_return = float(target_return_str.strip("%")) / 100
        else:
            self.target_return = target_return
        
        # 从strategy_config.yaml加载置信区间计算策略配置
        if confidence_strategy is None or strategy_params is None:
            strategy_config_path = Path(r"c:\Users\jimjiang\PycharmProjects\AlphaDog\strategy_config.yaml")
            if not strategy_config_path.exists():
                raise FileNotFoundError(f"未找到strategy_config.yaml，请检查路径：{strategy_config_path}")
            
            with open(strategy_config_path, "r", encoding="utf-8") as f:
                strategy_config = yaml.safe_load(f)
            
            # 获取策略名称和参数
            if confidence_strategy is None:
                confidence_strategy = strategy_config.get("confidence_strategy", "normal")
            
            if strategy_params is None:
                # 根据策略名称获取对应的参数
                if confidence_strategy == "normal":
                    strategy_params = strategy_config.get("normal_strategy_params", {})
                elif confidence_strategy == "quantile":
                    strategy_params = strategy_config.get("quantile_strategy_params", {})
                elif confidence_strategy == "yacd":
                    strategy_params = strategy_config.get("yacd_strategy_params", {})
                else:
                    strategy_params = {}
        
        # 创建置信区间计算策略
        strategy_params = strategy_params or {}
        self.confidence_strategy = StrategyFactory.get_strategy(confidence_strategy, **strategy_params)

    

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
        
        # 使用可插拔的置信区间计算策略
        lower_bound, upper_bound = self.confidence_strategy.calculate_bounds(df)
        
        # 记录使用的策略信息
        strategy_name = self.confidence_strategy.__class__.__name__
        print(f"使用置信区间计算策略: {strategy_name}")
        print(f"计算结果 - 下界: {lower_bound:.2f}, 上界: {upper_bound:.2f}")
        
        # 记录到history.md
        history_path = Path(r"c:\Users\jimjiang\PycharmProjects\AlphaDog\history.md")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(f"- [{timestamp}] 使用策略 {strategy_name} 计算置信区间: 下界={lower_bound:.2f}, 上界={upper_bound:.2f}\n")

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
    