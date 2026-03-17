import pandas as pd

# 1. 读取数据
file_path = r"PROCESS_DATA/#4峡沙56号_风机-激光雷达数据合并.csv"
df = pd.read_csv(file_path)
print(f"✅ 数据读取成功，共 {len(df)} 行。")

# 2. 转换时间列格式
df['DateAndTime'] = pd.to_datetime(df['DateAndTime'])

# 3. Distance == 0 时，HWS(hub)AVL 直接赋值为 100
df.loc[df['Distance'] == 0, 'HWS(hub)AVL'] = 100

# 4. 检查必需列
required_cols = [
    'DateAndTime', 'Distance',
    'RAWS', 'HWS(hub)', 'HWS(hub)AVL',
    'DIR(hub)', 'Veer', 'VShear', 'HShear',
    'TI1', 'TI2', 'TI3', 'TI4',
    '风机故障'
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"缺少以下列：{missing_cols}")

# 5. 找出满足条件的时间戳：
#    同一时间戳下所有 Distance 同时满足：
#    - HWS(hub)AVL > 60
#    - 风机故障 == 0
valid_times = df.groupby('DateAndTime').apply(
    lambda g: ((g['HWS(hub)AVL'] > 60) & (g['风机故障'] == 0)).all()
)

# 6. 保留这些时间戳下的所有记录
valid_df = df[df['DateAndTime'].isin(valid_times[valid_times].index)].copy()

# 7. 仅保留你需要的列
"""
    'DateAndTime',时间
    'Distance',距离
    'RAWS',轴线投影风速
    'HWS(hub)',反演风速
    'DIR(hub)',反演风向
    'Veer',垂直风向变化率
    'VShear',垂直风切变
    'HShear',水平风切变
    'TI1',光束编号1-湍流强度
    'TI2',光束编号2-湍流强度
    'TI3',光束编号3-湍流强度
    'TI4'光束编号4-湍流强度
"""
columns_to_keep = [
    'DateAndTime',
    'Distance',
    'RAWS',
    'HWS(hub)',
    'DIR(hub)',
    'Veer',
    'VShear',
    'HShear',
    'TI1',
    'TI2',
    'TI3',
    'TI4'
]
filtered_df = valid_df[columns_to_keep].copy()

# 8. 保存结果
output_csv_path = r"PROCESS_DATA\#7构建好的训练数据集.csv"
filtered_df.to_csv(output_csv_path, index=False)

# 9. 输出统计信息
total_time_count = df['DateAndTime'].nunique()
valid_time_count = int(valid_times.sum())
valid_time_ratio = valid_time_count / total_time_count if total_time_count > 0 else 0

print(f"✅ 已成功筛选并保存结果，共 {len(filtered_df)} 行。")
print(f"📁 文件保存路径：{output_csv_path}")
print(f"⏱️ 满足条件的时间戳数：{valid_time_count}")
print(f"⏱️ 总时间戳数：{total_time_count}")
print(f"📈 满足条件的时间戳比例：{valid_time_ratio:.4f}")