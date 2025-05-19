import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# 读取CSV文件
data = pd.read_csv('merged_cleaned_data.csv')

# 分离输入和输出
X = torch.tensor(data[['temperature', 'time']].values, dtype=torch.float32)  # 输入数据
y = torch.tensor(data['predict_T'].values, dtype=torch.float32).view(-1, 1)  # 输出数据

# 构建带ReLU的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.relu = nn.ReLU()  # 定义ReLU层
        
        # 自定义权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 使用正态分布初始化权重
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        # 初始化偏置为0
        nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)  # 正确使用ReLU

# 实例化模型
model = LinearRegressionModel()

# 打印初始化后的参数
print("初始化权重:", model.linear.weight)
print("初始化偏置:", model.linear.bias)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 输出预测结果
model.eval()
with torch.no_grad():
    predictions = model(X)
    print("Predictions:", predictions.flatten().numpy())
