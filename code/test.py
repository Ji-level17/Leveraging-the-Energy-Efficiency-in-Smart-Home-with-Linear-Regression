import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ======================
# 数据生成（示例用）
# ======================
def generate_arima_data(n=200, ar_params=[0.6, -0.2], ma_params=[0.3], d=1):
    """生成ARIMA(p,d,q)模拟数据"""
    # 生成基础ARMA序列
    p, q = len(ar_params), len(ma_params)
    errors = np.random.normal(0, 1, n + 50)  # 白噪声
    
    # 生成AR部分
    ar_data = np.zeros(n + 50)
    for t in range(p, n + 50):
        ar_data[t] = np.dot(ar_params, ar_data[t-p:t][::-1]) + errors[t]
    
    # 加入MA部分
    ma_data = np.zeros(n + 50)
    for t in range(q, n + 50):
        ma_data[t] = ar_data[t] + np.dot(ma_params, errors[t-q:t][::-1])
    
    # 差分处理
    diff_data = ma_data
    for _ in range(d):
        diff_data = np.diff(diff_data)
    
    # 截取平稳部分
    final_data = diff_data[50:50+n]
    return final_data

# 生成示例数据（ARIMA(2,1,1)）
np.random.seed(42)
original_data = generate_arima_data(ar_params=[0.6, -0.2], ma_params=[0.3], d=1)
time_index = pd.date_range(start='2023-01-01', periods=len(original_data), freq='D')

# ======================
# 核心ARIMA实现
# ======================
class SimpleARIMA:
    def __init__(self, p=1, d=1, q=1):
        self.p = p  # AR阶数
        self.d = d  # 差分阶数
        self.q = q  # MA阶数
        self.ar_coefs = None  # AR系数
        self.ma_coefs = None  # MA系数
        self.residuals = []   # 残差序列
    
    def difference(self, data, d):
        """执行d阶差分"""
        diff = data.copy()
        for _ in range(d):
            diff = np.diff(diff)
        return diff
    
    def prepare_lagged_matrix(self, data, p):
        """构建滞后矩阵用于AR模型"""
        n = len(data)
        X = np.zeros((n - p, p))
        for i in range(p, n):
            X[i - p] = data[i - p:i][::-1]  # 逆序获取 [t-1, t-2, ..., t-p]
        y = data[p:]
        return X, y
    
    def fit_ar(self, data):
        """使用最小二乘法估计AR系数"""
        X, y = self.prepare_lagged_matrix(data, self.p)
        self.ar_coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        return self
    
    def ma_residuals(self, data):
        """计算MA部分的残差"""
        predictions = np.zeros_like(data)
        residuals = np.zeros_like(data)
        for t in range(self.p, len(data)):
            if t < self.p:
                pred = 0
            else:
                pred = np.dot(self.ar_coefs, data[t - self.p:t][::-1])
            residuals[t] = data[t] - pred
        self.residuals = residuals[self.p:]
        return self
    
    def fit_ma(self, maxiter=100):
        """使用梯度下降法优化MA参数"""
        def loss_function(ma_params):
            # 计算MA预测误差
            predicted_errors = np.zeros(len(self.residuals))
            for t in range(self.q, len(self.residuals)):
                predicted_errors[t] = np.dot(ma_params, self.residuals[t - self.q:t][::-1])
            return np.mean((self.residuals[self.q:] - predicted_errors[self.q:])**2)
        
        # 初始猜测和优化
        initial_guess = np.zeros(self.q)
        result = minimize(loss_function, initial_guess, method='L-BFGS-B')
        self.ma_coefs = result.x
        return self
    
    def fit(self, data):
        """完整训练流程"""
        # 差分处理
        diff_data = self.difference(data, self.d)
        
        # 训练AR部分
        self.fit_ar(diff_data)
        
        # 计算残差（AR预测误差）
        self.ma_residuals(diff_data)
        
        # 训练MA部分（如果q>0）
        if self.q > 0:
            self.fit_ma()
        return self
    
    def forecast(self, data, steps=10):
        """执行多步预测"""
        history = list(data.copy())
        forecasts = []
        
        for _ in range(steps):
            # AR部分预测
            ar_pred = np.dot(self.ar_coefs, history[-self.p:][::-1])
            
            # MA部分预测（使用历史残差）
            if self.q > 0:
                ma_pred = np.dot(self.ma_coefs, self.residuals[-self.q:])
            else:
                ma_pred = 0
            
            # 综合预测
            combined_pred = ar_pred + ma_pred
            forecasts.append(combined_pred)
            
            # 更新历史数据（假设真实值未知时用预测值填充）
            history.append(combined_pred)
            self.residuals = np.append(self.residuals, 0)  # 假设新残差为0
        
        # 还原差分（示例仅处理d=1的情况）
        if self.d == 1:
            last_value = data[-1]
            restored = []
            for val in forecasts:
                last_value += val
                restored.append(last_value)
            return np.array(restored)
        else:
            return np.array(forecasts)

# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 初始化模型参数（与实际生成数据一致）00
    model = SimpleARIMA(p=2, d=1, q=1)
    
    # 训练模型
    model.fit(original_data)
    
    # 执行预测
    forecast_steps = 20
    predictions = model.forecast(original_data, steps=forecast_steps)
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original Data')
    plt.plot(range(len(original_data), len(original_data)+forecast_steps), 
             predictions, 'r--', label='Forecast')
    plt.title("Manual ARIMA Implementation")
    plt.legend()
    plt.show()
    
    # 输出模型参数
    print("AR Coefficients:", model.ar_coefs)
    print("MA Coefficients:", model.ma_coefs)