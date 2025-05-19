import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def load_and_preprocess(file_path, forecast_steps=20000):
    # 加载数据并划分训练测试集
    data = pd.read_csv(file_path, parse_dates=['created_at'], index_col='created_at')
    series = data['temperature']
    
    # 划分原始数据集
    train_original = series[:-forecast_steps]
    test_original = series[-forecast_steps:]
    
    # 仅对训练集做平稳性处理
    adf_result = adfuller(train_original)
    print(f'ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}')
    
    if adf_result[1] > 0.05:
        d = 1
        train_diff = train_original.diff(d).dropna()
        return train_diff, train_original, test_original, d
    else:
        d = 0
        return train_original, train_original, test_original, d

def arima_forecast(file_path, p, q):
    forecast_steps = 1000
    
    # 加载预处理数据
    train_diff, train_original, test_original, actual_d = load_and_preprocess(file_path)
    
    # 训练模型
    model = ARIMA(train_diff, order=(p, actual_d, q))
    model_fit = model.fit()
    
    # 预测差分值
    forecast_diff = model_fit.forecast(steps=forecast_steps)
    
    # 还原差分结果
    if d > 0:
        # 获取最后一个已知原始值
        last_original = train_original.iloc[-1]
        
        # 生成预测时间索引
        last_date = train_original.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(seconds=1),
            periods=forecast_steps,
            freq='S'
        )
        
        # 累加还原预测值
        restored = []
        current = last_original
        for val in forecast_diff:
            current += val
            restored.append(current)
        forecast = pd.Series(restored, index=forecast_dates)
    else:
        forecast = pd.Series(forecast_diff, index=test_original.index)
    
    # 计算MSE
    mse = np.mean((forecast - test_original) ** 2)
    print(f'MSE: {mse:.4f}')
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(train_original, label='Training Data')
    plt.plot(test_original, label='True Future Values', color='green')
    plt.plot(forecast, label='Forecast', linestyle='--', marker='o')
    plt.title(f'ARIMA({p},{actual_d},{q}) Forecast\nMSE: {mse:.2f}')
    plt.legend()
    plt.show()
    
    return forecast

if __name__ == "__main__":
    forecast = arima_forecast('merged_cleaned_data.csv', 5, 3)
    print("预测结果:\n", forecast)
