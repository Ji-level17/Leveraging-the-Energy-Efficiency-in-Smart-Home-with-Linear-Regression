import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb  # 或 import xgboost as xgb

# 数据加载和预处理
data = pd.read_csv('merged_cleaned_data.csv')

# 将时间列转换为datetime格式
data['time'] = pd.to_datetime(data['created_at'])

# ---------------------------
# 特征工程（关键修改部分）
# ---------------------------
# 时间特征分解
data['hour'] = data['time'].dt.hour
data['day_of_week'] = data['time'].dt.dayofweek
data['day_of_month'] = data['time'].dt.day

# 周期性编码
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

# 滞后特征（假设数据是按小时记录的）
lags = [1, 2, 3, 24, 24*7]  # 1小时、3小时、前一天、前一周
for lag in lags:
    data[f'temp_lag_{lag}'] = data['temperature'].shift(lag)

# 滚动特征
data['temp_rolling_mean_24h'] = data['temperature'].rolling(24).mean()
data['temp_rolling_std_24h'] = data['temperature'].rolling(24).std()

# 目标变量（假设需要预测下一时刻温度）
data['predict_T'] = data['temperature'].shift(-1)  # 下个时间点的温度

# 删除包含NaN的行（根据特征生成情况调整）
data.dropna(inplace=True)

# ---------------------------
# 数据集划分（时间序列专用方式）
# ---------------------------
# 按时间排序后划分
data = data.sort_values('time')
split_idx = int(len(data) * 0.8)  # 80%训练

train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

# 特征列选择（根据实际特征调整）
feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                'temp_lag_1', 'temp_lag_24', 'temp_rolling_mean_24h']

X_train = train[feature_cols]
y_train = train['predict_T']
X_test = test[feature_cols]
y_test = test['predict_T']

# ---------------------------
# LightGBM模型训练（可替换为XGBoost）
# ---------------------------
# LightGBM参数设置
params = {
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'verbosity': -1
}

# 创建数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 训练模型
model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# ---------------------------
# 预测与评估
# ---------------------------
y_pred = model.predict(X_test)

# 评估指标
print(f'MAE: {mean_absolute_error(y_test, y_pred):.4f}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')

# ---------------------------
# 可视化部分（保持与原始代码一致风格）
# ---------------------------
# 实际vs预测散点图
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], 
         [min(y_test), max(y_test)], 
         'r--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost/LightGBM Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# 时间序列趋势图
plt.figure(figsize=(12, 6))
plt.plot(test['time'], y_test.values, label='Actual', alpha=0.8)
plt.plot(test['time'], y_pred, label='Predicted', alpha=0.8, linestyle='--')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('XGBoost/LightGBM Temperature Prediction over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 特征重要性可视化
lgb.plot_importance(model, max_num_features=10, figsize=(10, 6))
plt.title('Feature Importance')
plt.show()