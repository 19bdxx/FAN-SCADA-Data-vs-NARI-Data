import pandas as pd
import numpy as np


def circular_mean_deg(series):
    """
    计算角度序列（单位：度，范围 0~360）的环形平均值。
    忽略 NaN；若全为空则返回 NaN。
    """
    series = series.dropna()
    if len(series) == 0:
        return np.nan

    radians = np.deg2rad(series % 360)
    mean_sin = np.sin(radians).mean()
    mean_cos = np.cos(radians).mean()

    # 若均值向量接近 0，atan2 结果虽可算，但物理意义弱
    if np.isclose(mean_sin, 0) and np.isclose(mean_cos, 0):
        return np.nan

    angle = np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360
    return angle


def circular_resultant_length(series):
    """
    计算角度序列（单位：度，范围 0~360）的平均向量长度 R。
    R 取值范围 0~1：
    - 越接近 1，方向越集中
    - 越接近 0，方向越分散
    """
    series = series.dropna()
    if len(series) == 0:
        return np.nan

    radians = np.deg2rad(series % 360)
    mean_sin = np.sin(radians).mean()
    mean_cos = np.cos(radians).mean()

    r = np.hypot(mean_sin, mean_cos)
    return r


def process_wind_power_data(input_file_path, output_file_path, r_threshold=0.1):
    """
    处理风机1分钟数据，生成10分钟粒度版本，并打印缺失数据的时间段：

    规则：
    1. 仅保留前10分钟数据完整的时间段，不完整时间段均值记为 NaN
    2. 功率、风速使用普通平均
    3. 风向使用环形平均
    4. 若风向平均向量长度 R < r_threshold，则认为该窗口风向过于分散，
       风向均值记为 NaN
    """

    # 读取数据
    df = pd.read_csv(input_file_path, encoding='utf-8-sig')
    df['时间'] = pd.to_datetime(df['时间'])

    if df['时间'].duplicated().any():
        dup_times = df.loc[df['时间'].duplicated(keep=False), '时间'].sort_values()
        raise ValueError(f"输入数据存在重复时间戳，请先检查对齐结果。例如：\n{dup_times.head(20)}")

    # 检查必需列
    required_cols = [
        '时间',
        '平均有功功率_风机导出',
        'ACTIVE_POWER_#56_对齐',
        'WINDSPEED_#56_对齐',
        'WINDDIRECTION_#56_对齐'
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"输入文件缺少以下列：{missing_cols}")

    # 取需要的列
    df_numeric = df[required_cols].copy()
    df_numeric.set_index('时间', inplace=True)

    # 10分钟计数（完整性检查）
    count_per_bin = df_numeric.resample('10min', label='right', closed='right').count()

    # 普通均值列：功率、风速
    normal_cols = [
        '平均有功功率_风机导出',
        'ACTIVE_POWER_#56_对齐',
        'WINDSPEED_#56_对齐'
    ]
    mean_normal = df_numeric[normal_cols].resample(
        '10min', label='right', closed='right'
    ).mean()

    # 风向环形均值
    mean_direction = df_numeric['WINDDIRECTION_#56_对齐'].resample(
        '10min', label='right', closed='right'
    ).apply(circular_mean_deg)

    # 风向集中度 R
    direction_r = df_numeric['WINDDIRECTION_#56_对齐'].resample(
        '10min', label='right', closed='right'
    ).apply(circular_resultant_length)

    # 合并结果
    mean_per_bin = mean_normal.copy()
    mean_per_bin['WINDDIRECTION_#56_对齐'] = mean_direction
    mean_per_bin['WINDDIRECTION_#56_对齐_R'] = direction_r

    # 完整性检查：四列都必须满 10 条
    check_cols = [
        '平均有功功率_风机导出',
        'ACTIVE_POWER_#56_对齐',
        'WINDSPEED_#56_对齐',
        'WINDDIRECTION_#56_对齐'
    ]
    incomplete_mask = count_per_bin[check_cols].min(axis=1) < 10
    missing_times = count_per_bin.index[incomplete_mask]

    if not missing_times.empty:
        print("以下时间点对应的前10分钟数据不完整，均值结果为 NaN：")
        for ts in missing_times:
            print(ts.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print("所有10分钟时间段数据完整，无缺失。")

    # 风向分散度检查
    low_r_mask = mean_per_bin['WINDDIRECTION_#56_对齐_R'] < r_threshold
    low_r_times = mean_per_bin.index[low_r_mask.fillna(False)]

    if len(low_r_times) > 0:
        print(f"\n以下时间点对应的前10分钟风向过于分散（R < {r_threshold}），风向均值记为 NaN：")
        for ts in low_r_times:
            print(ts.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print(f"\n所有10分钟时间段风向集中度均满足 R >= {r_threshold}。")

    # 对低 R 的窗口，仅将风向均值置为 NaN
    mean_per_bin.loc[low_r_mask, 'WINDDIRECTION_#56_对齐'] = pd.NA

    # 对不完整窗口，整行都置为 NaN
    mean_per_bin.loc[incomplete_mask, :] = pd.NA

    # 重命名列
    result = mean_per_bin.reset_index().rename(columns={
        '平均有功功率_风机导出': '平均有功功率_风机导出_前10分钟均值',
        'ACTIVE_POWER_#56_对齐': 'ACTIVE_POWER_#56_对齐_前10分钟均值',
        'WINDSPEED_#56_对齐': 'WINDSPEED_#56_对齐_前10分钟均值',
        'WINDDIRECTION_#56_对齐': 'WINDDIRECTION_#56_对齐_前10分钟环形均值',
        'WINDDIRECTION_#56_对齐_R': 'WINDDIRECTION_#56_对齐_前10分钟集中度R'
    })

    # 保存
    result.to_csv(output_file_path, index=False, encoding='gbk')
    return result


if __name__ == "__main__":
    input_file = r'PROCESS_DATA\#2-1对齐风机导出数据和南瑞数据.csv'
    output_file = r'PROCESS_DATA\#3峡沙56号_时间戳对齐后数据_10分钟均值.csv'

    result = process_wind_power_data(
        input_file_path=input_file,
        output_file_path=output_file,
        r_threshold=0.1
    )

    print(f"\n处理完成，结果已保存至 {output_file}")