import pandas as pd
import glob

# 方式1：自动获取目录下所有CSV文件（根据需求选择其中一种方式）
# 请将路径替换为你的CSV文件所在目录
# file_list = glob.glob('./path_to_your_csv_files/*.csv')  # 匹配所有csv文件

# 方式2：手动指定文件列表（根据需求选择其中一种方式）
file_list = [
    'feeds_canteen1.csv',
    'feeds_livingroom.csv',
    'feeds_studyroom.csv'
]

# 存储所有DataFrame的列表
df_list = []

# 读取每个CSV文件
for file in file_list:
    try:
        # 读取CSV文件，根据实际情况可能需要指定编码格式（如encoding='gbk'）
        df = pd.read_csv(file)
        df_list.append(df)
        print(f'已成功加载: {file}')
    except Exception as e:
        print(f'加载 {file} 失败，错误信息: {str(e)}')

# 纵向合并所有DataFrame
if df_list:
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    # 保存合并后的结果
    merged_df.to_csv('merged_result.csv', index=False)
    print('合并完成，结果已保存为 merged_result.csv')
else:
    print('没有找到可合并的CSV文件')
