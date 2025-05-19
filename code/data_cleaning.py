import pandas as pd
import os
# 读取CSV文件（假设列名包含'temperature', 'humidity', 'weather_label'）
df = pd.read_csv('feeds_canteen1.csv')
# 选择需要归一化的列
cols_to_scale = ["humidity","light","temperature","time"]

# 计算最小值和范围
data_min = df[cols_to_scale].min()
data_range = df[cols_to_scale].max() - data_min

# 应用归一化公式：(x - min)/(max - min)
df_scaled = df.copy()
df_scaled[cols_to_scale] = (df[cols_to_scale] - data_min) / data_range

# 保存处理后的数据
df_scaled.to_csv('merged_cleaned_data.csv', index=False)


# length = len(df)
# i = 0
# # print(df.dtypes)
# while(i<length):
#     df.loc[i]["time"] = df.loc[i]["time"] / 60
#     i+=1

# os.makedirs(os.path.dirname('feeds_canteen1.csv'), exist_ok=True)
# df.to_csv('feeds_canteen1.csv', index=False, encoding='utf-8-sig')  # 兼容Excel中文
# print(f"数据已保存至 {'feeds_canteen1.csv'}")
