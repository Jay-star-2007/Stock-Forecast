import numpy as np
import pandas as pd

class PerformanceCalculator:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        
    def calculate_returns(self, actual_prices, predicted_prices, transaction_cost=0.001):
        """计算模型预测的收益率"""
        capital = self.initial_capital
        position = 0  # 0表示空仓，1表示持仓
        trades = []
        daily_returns = []
        
        # 设置交易限制
        max_trades_per_day = 1  # 每天最多交易一次
        min_holding_period = 5  # 最小持仓周期（天）
        last_trade_day = -min_holding_period  # 上次交易日
        stop_loss = -0.05  # 止损线（-5%）
        take_profit = 0.10  # 止盈线（10%）
        
        entry_price = 0  # 入场价格
        
        for i in range(1, len(actual_prices)):
            current_price = actual_prices[i]
            current_return = 0
            
            # 计算当前持仓收益率
            if position == 1:
                current_return = (current_price - entry_price) / entry_price
            
            # 止损止盈检查
            if position == 1 and (current_return <= stop_loss or current_return >= take_profit):
                position = 0
                capital *= (1 + current_return)
                capital *= (1 - transaction_cost)
                trades.append(('止损止盈', current_price, i))
                last_trade_day = i
            
            # 正常交易信号
            elif i - last_trade_day >= min_holding_period:  # 确保满足最小持仓周期
                pred_return = (predicted_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
                
                # 设置信号阈值，避免过于频繁交易
                buy_threshold = 0.01   # 1%的预期收益才买入
                sell_threshold = -0.01 # -1%的预期收益才卖出
                
                if pred_return > buy_threshold and position == 0:  # 买入信号
                    position = 1
                    entry_price = current_price
                    capital *= (1 - transaction_cost)
                    trades.append(('买入', current_price, i))
                    last_trade_day = i
                elif pred_return < sell_threshold and position == 1:  # 卖出信号
                    position = 0
                    capital *= (1 + current_return)
                    capital *= (1 - transaction_cost)
                    trades.append(('卖出', current_price, i))
                    last_trade_day = i
            
            # 记录当日收益
            daily_return = ((current_price - actual_prices[i-1]) / actual_prices[i-1]) if position == 1 else 0
            daily_returns.append(daily_return)
        
        # 计算年化收益率
        total_days = len(actual_prices)
        annual_return = ((capital / self.initial_capital) ** (252/total_days) - 1) * 100
        
        # 计算其他指标
        daily_returns = np.array(daily_returns)
        sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 0 else 0
        max_drawdown = self._calculate_max_drawdown(daily_returns)
        
        return {
            'total_return': ((capital - self.initial_capital) / self.initial_capital) * 100,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'trade_count': len(trades),
            'final_capital': capital,
            'trades': trades
        }
    
    def _calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        return np.max(drawdowns) if len(drawdowns) > 0 else 0