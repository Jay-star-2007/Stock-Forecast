import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

class StockRFPredictor:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        
    def prepare_data(self, data):
        # 处理无效数据
        data = np.nan_to_num(data, nan=np.nanmean(data))
        
        # 对数变换来处理价格数据
        data = np.log1p(data)
        
        # 标准化数据
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i:(i + self.look_back), 0])
            y.append(scaled_data[i + self.look_back, 0])
            
        return np.array(X), np.array(y)
    
    def train(self, data):
        X, y = self.prepare_data(data)
        self.model.fit(X, y)
        
    def predict(self, data):
        X, _ = self.prepare_data(data)
        predictions = self.model.predict(X)
        predictions = predictions.reshape(-1, 1)
        
        # 反转标准化和对数变换
        predictions = self.scaler.inverse_transform(predictions)
        predictions = np.expm1(predictions)
        
        return predictions 