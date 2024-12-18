from flask import Flask, render_template, jsonify
import pandas as pd
from model.lstm_model import StockVolumePredictor
from model.rf_model import StockRFPredictor
import numpy as np
import os
from utils.performance import PerformanceCalculator

app = Flask(__name__)

def load_stock_data(year='1990'):
    """加载股票数据和SP500成分股数据"""
    try:
        # 加载收盘价和开盘价数据
        close_path = os.path.join('data', f'Close-{year}.csv')
        open_path = os.path.join('data', f'Open-{year}.csv')
        spx_path = 'data/SPXconst.csv'
        
        # 检查文件是否存在
        for path in [close_path, open_path, spx_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f'数据文件不存在: {path}')
        
        # 读取数据
        close_df = pd.read_csv(close_path)
        open_df = pd.read_csv(open_path)
        spx_const = pd.read_csv(spx_path)
        
        # 数据清理
        # 1. 确保日期列存在并转换格式
        if 'Date' not in close_df.columns or 'Date' not in open_df.columns:
            raise ValueError('数据文件缺少日期列')
            
        # 转换日期格式
        close_df['Date'] = pd.to_datetime(close_df['Date']).dt.strftime('%Y-%m-%d')
        open_df['Date'] = pd.to_datetime(open_df['Date']).dt.strftime('%Y-%m-%d')
        
        # 2. 获取有效的股票代码
        close_stocks = set(close_df.columns) - {'Date'}
        open_stocks = set(open_df.columns) - {'Date'}
        spx_stocks = set(spx_const.iloc[0].dropna().tolist())
        
        # 3. 找出所有数据集中都存在的股票
        valid_stocks = close_stocks.intersection(open_stocks, spx_stocks)
        
        if not valid_stocks:
            raise ValueError('没有找到有效的股票数据')
            
        # 4. 只保留有效的股票数据
        valid_columns = ['Date'] + list(valid_stocks)
        close_df = close_df[valid_columns]
        open_df = open_df[valid_columns]
        
        # 5. 检查数据完整性（添加详细诊断）
        stocks_to_remove = set()
        for stock in valid_stocks:
            # 详细诊断信息
            close_nulls = close_df[stock].isnull().sum()
            open_nulls = open_df[stock].isnull().sum()
            close_zeros = (close_df[stock] <= 0).sum()
            open_zeros = (open_df[stock] <= 0).sum()
            
            # 打印详细的诊断信息
            if close_nulls > 0 or open_nulls > 0 or close_zeros > 0 or open_zeros > 0:
                print(f'股票 {stock} 数据诊断:')
                print(f'  - 收盘价缺失值数量: {close_nulls}')
                print(f'  - 开盘价缺失值数量: {open_nulls}')
                print(f'  - 收盘价异常值数量: {close_zeros}')
                print(f'  - 开盘价异常值数量: {open_zeros}')
            
            # 检查缺失值
            null_count = close_nulls + open_nulls
            if null_count > len(close_df) * 0.1:  # 超过10%的数据缺失
                print(f'警告: 股票 {stock} 缺失数据过多 ({null_count}条)，将被移除')
                stocks_to_remove.add(stock)
            # 检查异常值
            elif close_zeros > 0 or open_zeros > 0:
                print(f'警告: 股票 {stock} 存在异常数据 (零或负值)，将被移除')
                stocks_to_remove.add(stock)
            
            # 添加数据范围检查
            if stock not in stocks_to_remove:
                close_range = close_df[stock].describe()
                open_range = open_df[stock].describe()
                print(f'股票 {stock} 数据范围:')
                print(f'  收盘价: {close_range["min"]:.2f} - {close_range["max"]:.2f}')
                print(f'  开盘价: {open_range["min"]:.2f} - {open_range["max"]:.2f}')
        
        # 移除问题股票
        if stocks_to_remove:
            valid_columns = ['Date'] + list(valid_stocks - stocks_to_remove)
            close_df = close_df[valid_columns]
            open_df = open_df[valid_columns]
        
        # 6. 按日期排序
        close_df = close_df.sort_values('Date')
        open_df = open_df.sort_values('Date')
        
        # 重置索引
        close_df = close_df.reset_index(drop=True)
        open_df = open_df.reset_index(drop=True)
        
        print(f'数据加载完成。可用股票数量: {len(close_df.columns) - 1}')
        return close_df, open_df, spx_const
        
    except Exception as e:
        print(f'加载数据时出错: {str(e)}')
        raise

def prepare_training_data(close_df, open_df, spx_const, stock_symbol=None):
    """准备训练数据
    
    Args:
        close_df: 收盘价数据
        open_df: 开盘价数据
        spx_const: SP500成分股数据
        stock_symbol: 指定股票代码，如果为None则使用第一个可用的股票
    """
    if stock_symbol is None:
        # 获取第一个在SP500成分股中的股票
        available_stocks = set(close_df.columns[1:]).intersection(spx_const.iloc[0].dropna())
        stock_symbol = list(available_stocks)[0]
    
    # 构建训练数据
    training_data = {
        'date': close_df['Date'],
        'close': close_df[stock_symbol],
        'open': open_df[stock_symbol]
    }
    
    return pd.DataFrame(training_data), stock_symbol

# 加载数据和训练模型
def load_and_train_model():
    close_df, open_df, spx_const = load_stock_data()
    stock_data, _ = prepare_training_data(close_df, open_df, spx_const)
    
    # 训练LSTM模型
    lstm_predictor = StockVolumePredictor()
    lstm_predictor.train(stock_data['close'].values)
    
    # 训练RandomForest模型
    rf_predictor = StockRFPredictor()
    rf_predictor.train(stock_data['close'].values)
    
    return lstm_predictor, rf_predictor, stock_data, open_df

lstm_predictor, rf_predictor, stock_data, open_df = load_and_train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stocks')
def get_available_stocks():
    """获取可用的股票列表"""
    try:
        close_df, open_df, _ = load_stock_data()
        
        # 获取所有可用的股票代码（排除Date列）
        available_stocks = sorted(set(close_df.columns) - {'Date'})
        
        if not available_stocks:
            return jsonify({'error': '没有找到可用的股票数据'}), 404
            
        # 进一步验证每个股票的数据
        valid_stocks = []
        for stock in available_stocks:
            # 检查数据长度
            if len(close_df[stock]) >= 60:  # 确保有足够的数据进行预测
                # 检查数据质量
                if not (close_df[stock].isnull().any() or open_df[stock].isnull().any()):
                    valid_stocks.append(stock)
                else:
                    print(f'警告: 股票 {stock} 包含无效数据，已被过滤')
        
        print(f'有效股票数量: {len(valid_stocks)}')
        return jsonify(valid_stocks)
        
    except Exception as e:
        print(f'获取股票列表失败: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<stock_symbol>')
def get_prediction(stock_symbol):
    """获取特定股票的预测结果"""
    try:
        close_df, open_df, spx_const = load_stock_data()
        
        # 验证股票代码是否存在
        if stock_symbol not in close_df.columns or stock_symbol not in open_df.columns:
            return jsonify({'error': f'股票代码 {stock_symbol} 数据不完整'}), 404
            
        # 验证数据是否有效
        close_data = close_df[stock_symbol].values
        open_data = open_df[stock_symbol].values
        
        # 检查是否存在无效数据
        if np.isnan(close_data).any() or np.isnan(open_data).any():
            return jsonify({'error': f'股票 {stock_symbol} 包含无效数据'}), 400
            
        # 检查数据长度是否足够
        if len(close_data) < 60:  # 假设我们需要至少60天的数据
            return jsonify({'error': f'股票 {stock_symbol} 数据量不足'}), 400
            
        stock_data, _ = prepare_training_data(close_df, open_df, spx_const, stock_symbol)
        
        # 计算日涨跌幅
        try:
            stock_data['change_rate'] = ((stock_data['close'] - stock_data['open']) / stock_data['open'] * 100)
        except Exception as e:
            return jsonify({'error': f'计算涨跌幅失败: {str(e)}'}), 500
        
        # 使用两个模型进行预测
        try:
            training_data = stock_data['close'].values
            lstm_predictions = lstm_predictor.predict(training_data)
            rf_predictions = rf_predictor.predict(training_data)
            
            # 验证预测结果
            if np.isnan(lstm_predictions).any() or np.isnan(rf_predictions).any():
                return jsonify({'error': '模型预测结果包含无效数据'}), 500
                
        except Exception as e:
            return jsonify({'error': f'预测失败: {str(e)}'}), 500
            
        # 计算性能指标
        try:
            calculator = PerformanceCalculator()
            lstm_performance = calculator.calculate_returns(
                training_data[-len(lstm_predictions):],
                lstm_predictions.flatten()
            )
            rf_performance = calculator.calculate_returns(
                training_data[-len(rf_predictions):],
                rf_predictions.flatten()
            )
        except Exception as e:
            return jsonify({'error': f'计算性能指标失败: {str(e)}'}), 500
        
        return jsonify({
            'dates': stock_data['date'].tolist(),
            'actual': training_data[-len(lstm_predictions):].tolist(),
            'lstm_predicted': lstm_predictions.flatten().tolist(),
            'rf_predicted': rf_predictions.flatten().tolist(),
            'open': stock_data['open'][-len(lstm_predictions):].tolist(),
            'change_rate': stock_data['change_rate'][-len(lstm_predictions):].tolist(),
            'performance': {
                'lstm': lstm_performance,
                'rf': rf_performance
            }
        })
    except Exception as e:
        return jsonify({'error': f'处理请求时发生错误: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 