import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ------------------------------
# 1. 加载CSV数据（请根据实际文件路径和列名修改）
# ------------------------------
# 假设CSV包含两列：timestamp（秒级时间戳）、temperature（温度值）
df = pd.read_csv("merged_cleaned_data.csv")  # 替换为实际文件路径

# 检查数据格式
print("原始数据样例：\n", df.head(3))

# ------------------------------
# 2. 数据预处理
# ------------------------------
# 转换为时间序列索引（假设时间戳为数值型秒数）
df['time'] = pd.to_datetime(df['time'], unit='s')  # 如果时间戳是Unix秒
df.set_index('time', inplace=True)

# 排序确保时间顺序
df = df.sort_index()

# ------------------------------
# 3. 特征工程：构建预测目标（未来1小时温度变化）
# ------------------------------
# 定义预测跨度（3600秒=1小时）
PREDICTION_STEP = 3600

# 创建目标变量：未来3600秒后的温度变化（温度差）
df['target'] = df['temperature'].shift(-PREDICTION_STEP) - df['temperature']

# ------------------------------
# 4. 构建滞后特征（根据时间窗口自定义）
# ------------------------------
def create_lag_features(df, window_seconds=[60, 300, 3600]):
    """创建基于不同时间窗口的滞后特征"""
    
    for window in window_seconds:
        # 计算窗口内的统计特征
        df[f'mean_{window}s'] = df['temperature'].rolling(f"{window}s").mean()
        df[f'min_{window}s'] = df['temperature'].rolling(f"{window}s").min()
        df[f'max_{window}s'] = df['temperature'].rolling(f"{window}s").max()
        df[f'std_{window}s'] = df['temperature'].rolling(f"{window}s").std()
    
    # 添加时间特征（周期性编码）
    seconds_in_day = 24 * 60 * 60
    df['sin_time'] = np.sin(2 * np.pi * df.index.hour/24)  # 日周期
    df['cos_time'] = np.cos(2 * np.pi * df.index.hour/24)
    
    # 删除包含NaN的行（由滚动窗口和未来目标生成）
    df.dropna(inplace=True)
    return df

# 使用时间窗口：[1分钟(60s), 5分钟(300s), 1小时(3600s)]
df = create_lag_features(df, window_seconds=[60, 300, 3600])
print("\n处理后的数据样例:\n", df.head(3))

# ------------------------------
# 5. 划分数据集（按时间顺序）
# ------------------------------
split_time = df.index[int(len(df)*0.8)]  # 前80%训练，后20%测试
train = df[df.index < split_time]
test = df[df.index >= split_time]

X_train = train.drop(['time', 'temperature', 'predict_T'], axis=1, errors='ignore')
y_train = train['predict_T']
X_test = test.drop(['time', 'temperature', 'predict_T'], axis=1, errors='ignore')
y_test = test['predict_T']

# ------------------------------
# 6. 训练随机森林模型
# ------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# ------------------------------
# 7. 模型评估
# ------------------------------
y_pred = model.predict(X_test)

print("\n=== 评估结果 ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# ------------------------------
# 8. 可视化预测效果（显示最近3天的预测）
# ------------------------------
plot_df = test.iloc[-3*3600:]  # 取最后3天的数据

plt.figure(figsize=(15, 6))
plt.plot(plot_df.index, plot_df['target'], label="实际变化", alpha=0.7)
plt.plot(plot_df.index, model.predict(X_test.loc[plot_df.index]), 
         label="预测变化", linestyle='--', alpha=0.9)
plt.title("实际 vs 预测温度变化(未来1小时)")
plt.xlabel("时间")
plt.ylabel("温度变化 (°C)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 9. 实时预测函数
# ------------------------------
def predict_future_change(model, latest_data, feature_columns):
    """输入最新时序数据,预测未来1小时温度变化
    参数：
        latest_data: DataFrame 包含最新时间窗口的原始数据
        feature_columns: 模型使用的特征列名列表
    返回：
        预测的温度变化值
    """
    # 生成最新特征
    latest_with_features = create_lag_features(latest_data.copy(), window_seconds=[60, 300, 3600])
    
    # 提取最后有效行
    input_features = latest_with_features[feature_columns].iloc[[-1]]
    
    return model.predict(input_features)[0]

# 示例用法：
# 假设需要保留足够的历史数据来生成特征（至少3600秒）
historical_data = df.last("2h")  # 保留最近2小时数据
current_change = predict_future_change(model, historical_data, X_train.columns)
print(f"\n预测未来1小时温度变化:{current_change:.2f}°C")