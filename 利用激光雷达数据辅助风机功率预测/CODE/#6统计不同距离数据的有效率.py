import pandas as pd

# 1. 读取CSV文件
file_path = r"PROCESS_DATA\#4峡沙56号_风机-激光雷达数据合并.csv"
df = pd.read_csv(file_path)

# 2. 时间列转换
df['DateAndTime'] = pd.to_datetime(df['DateAndTime'])

# 3. Distance==0 时，HWS(hub)AVL 直接赋值为 100
df.loc[df['Distance'] == 0, 'HWS(hub)AVL'] = 100

# 3. 状态列（按你现在的新逻辑）
status_col = '风机故障'

if status_col not in df.columns:
    raise KeyError(f"找不到列：{status_col}，请先检查当前文件字段名。")

# 4. 每个时间戳是否所有 Distance 的 HWS(hub)AVL > 60
all_above_60_per_time = df.groupby('DateAndTime')['HWS(hub)AVL'].apply(lambda x: (x > 60).all())

overall_time_ratio = all_above_60_per_time.mean()
print(f"\n✅ 所有距离在同一时间戳下 HWS(hub)AVL 全都 > 60 的比例为：{overall_time_ratio:.4f}")

not_all_above_60 = all_above_60_per_time[~all_above_60_per_time].index
not_all_above_60_df = pd.DataFrame(not_all_above_60, columns=['DateAndTime'])
output_path = r"PROCESS_DATA\#6不是都大于60的时间戳.csv"
not_all_above_60_df.to_csv(output_path, index=False)
print("\n✅ 已导出不是所有距离都大于60的时间戳：")
print(f"\n文件已保存至：{output_path}")

# 5. 每个 Distance 下 > 60 的比例
ratio_per_distance = df.groupby('Distance')['HWS(hub)AVL'].apply(lambda x: (x > 60).mean())
print("\n📊 每个 Distance 下 HWS(hub)AVL > 60 的比例：")
print(ratio_per_distance)

# 6. 每个 Distance 下，非故障（风机故障=0）的比例
non_fault_ratio_per_distance = df.groupby('Distance')[status_col].apply(lambda x: (x == 0).mean())
print("\n📊 每个 Distance 下 非故障（风机故障=0）的比例：")
print(non_fault_ratio_per_distance)

# 7. 每个 Distance 下，HWS(hub)AVL > 60 且 非故障（风机故障=0）的比例
condition_ratio_per_distance = df.groupby('Distance').apply(
    lambda g: ((g['HWS(hub)AVL'] > 60) & (g[status_col] == 0)).mean()
)
print("\n📊 每个 Distance 下 HWS(hub)AVL > 60 且 非故障（风机故障=0）的比例：")
print(condition_ratio_per_distance)

# 8. 所有 Distance 合并后，HWS(hub)AVL > 60 且 非故障（风机故障=0）的整体比例
total_condition_ratio = ((df['HWS(hub)AVL'] > 60) & (df[status_col] == 0)).mean()
print(f"\n📊 所有 Distance 下 HWS(hub)AVL > 60 且 非故障（风机故障=0）的总体比例：{total_condition_ratio:.4f}")

# 9. 每个时间戳是否所有 Distance 同时满足：HWS(hub)AVL > 60 且 非故障
full_condition_per_time = df.groupby('DateAndTime').apply(
    lambda g: ((g['HWS(hub)AVL'] > 60) & (g[status_col] == 0)).all()
)

full_condition_ratio = full_condition_per_time.mean()
print(f"\n✅ 所有距离在同一时间戳下 HWS(hub)AVL > 60 且 非故障（风机故障=0）的比例为：{full_condition_ratio:.4f}")

not_full_ok = full_condition_per_time[~full_condition_per_time].index
not_full_ok_df = pd.DataFrame(not_full_ok, columns=['DateAndTime'])
output_path_condition = r"PROCESS_DATA\#6不是所有距离HWS大于60且非故障的时间戳.csv"
not_full_ok_df.to_csv(output_path_condition, index=False)
print(f"\n📁 已保存不满足【HWS>60 且 非故障】的时间戳至：{output_path_condition}")