import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf

class StockVolumePredictor:
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        
    def _build_model(self):
        # 使用tf.keras.regularizers来防止过拟合
        model = Sequential([
            LSTM(50, return_sequences=True, 
                 input_shape=(self.look_back, 1),
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.2),
            LSTM(50, return_sequences=False,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        # 使用Huber损失函数来处理异常值
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
        return model
    
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
    
    def train(self, data, epochs=50, batch_size=32):
        X, y = self.prepare_data(data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # 添加早停机制
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
    def predict(self, data):
        X, _ = self.prepare_data(data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        predictions = self.model.predict(X)
        
        # 反转标准化和对数变换
        predictions = self.scaler.inverse_transform(predictions)
        predictions = np.expm1(predictions)
        
        return predictions 