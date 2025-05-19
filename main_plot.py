import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # 添加matplotlib库

# get csv file
data = pd.read_csv('merged_cleaned_data.csv')

# seperate input and output
X = torch.tensor(data[['temperature', 'time']].values, dtype=torch.float32)
y = torch.tensor(data['predict_T'].values, dtype=torch.float32).view(-1, 1)

# linear regression with Relu
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        return self.relu(self.linear(x))

# model
model = LinearRegressionModel()

# define the loss function and gradient down
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# trainning model
epochs = 100
losses = []  

for epoch in range(epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())  # recorded the losses
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# the actual vs prediction
model.eval()
with torch.no_grad():
    predictions = model(X)
    predictions = predictions.numpy().flatten()
    actual_values = y.numpy().flatten()

plt.figure(figsize=(10, 5))
plt.scatter(actual_values, predictions, alpha=0.5, label='Predicted vs Actual')
plt.plot([min(actual_values), max(actual_values)], 
         [min(actual_values), max(actual_values)], 
         'r--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# plot the figure feature vs time
time_values = X[:, 1].numpy()
plt.figure(figsize=(12, 6))
plt.scatter(time_values*24, actual_values, label='Actual', alpha=0.5)
plt.scatter(time_values*24, predictions, label='Predicted', alpha=0.5, marker='x')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Prediction over Time')
plt.legend()
plt.grid(True)
plt.show()