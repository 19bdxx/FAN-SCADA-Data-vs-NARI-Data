import pandas as pd


def add_fault_status_for_10min_window(
    df: pd.DataFrame,
    time_col: str,
    fault_df: pd.DataFrame,
    start_col: str = 'start_time',
    end_col: str = 'end_time'
) -> pd.Series:
    """
    对10分钟粒度数据添加“风机故障”状态列。

    约定：
    - df[time_col] 是10分钟窗口的右端点时间 T
    - 每条记录对应的窗口为 (T-10min, T]
    - 若该窗口与任一故障区间 [start_time, end_time] 有交集，则记为 1，否则为 0
    """
    fault_status = pd.Series(0, index=df.index, dtype=int)

    window_end = df[time_col]
    window_start = window_end - pd.Timedelta(minutes=10)

    for _, row in fault_df.iterrows():
        fault_start = row[start_col]
        fault_end = row[end_col]

        if pd.isna(fault_start) or pd.isna(fault_end):
            continue

        # 判断 (window_start, window_end] 与 [fault_start, fault_end] 是否有交集
        mask = (fault_end > window_start) & (fault_start <= window_end)
        fault_status.loc[mask] = 1

    return fault_status


def main():
    # ==============================
    # 1. 文件路径
    # ==============================
    file1 = r'PROCESS_DATA\#3峡沙56号_时间戳对齐后数据_10分钟均值.csv'
    file2 = r'RAW_DATA\合并后的风机雷达数据.csv'
    file3 = r'RAW_DATA\#56风机故障时刻.csv'

    output_file = r'PROCESS_DATA\#4峡沙56号_风机-激光雷达数据合并.csv'

    # ==============================
    # 2. 读取文件
    # ==============================
    print('[1/5] 读取文件...')

    # 文件1：10分钟均值文件
    df1 = pd.read_csv(file1, encoding='gbk')
    df1['时间'] = pd.to_datetime(df1['时间'])

    # 文件2：风机雷达数据
    try:
        df2 = pd.read_csv(file2, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df2 = pd.read_csv(file2, encoding='gbk')
    df2['DateAndTime'] = pd.to_datetime(df2['DateAndTime'])

    # 文件3：故障时刻
    try:
        df3 = pd.read_csv(file3, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df3 = pd.read_csv(file3, encoding='gbk')
    df3['start_time'] = pd.to_datetime(df3['start_time'])
    df3['end_time'] = pd.to_datetime(df3['end_time'])

    print(f'    文件1行数: {len(df1):,}')
    print(f'    文件2行数: {len(df2):,}')
    print(f'    文件3行数: {len(df3):,}')

    # ==============================
    # 3. 检查字段
    # ==============================
    print('[2/5] 检查字段...')

    required_cols_df1 = [
        '时间',
        '平均有功功率_风机导出_前10分钟均值',
        'ACTIVE_POWER_#56_对齐_前10分钟均值',
        'WINDSPEED_#56_对齐_前10分钟均值',
        'WINDDIRECTION_#56_对齐_前10分钟环形均值'
    ]
    required_cols_df2 = [
        'DateAndTime',
        'Distance',
        'HWS(hub)',
        'DIR(hub)'
    ]
    required_cols_df3 = [
        'start_time',
        'end_time'
    ]

    missing1 = [c for c in required_cols_df1 if c not in df1.columns]
    missing2 = [c for c in required_cols_df2 if c not in df2.columns]
    missing3 = [c for c in required_cols_df3 if c not in df3.columns]

    if missing1:
        raise KeyError(f'文件1缺少列: {missing1}')
    if missing2:
        raise KeyError(f'文件2缺少列: {missing2}')
    if missing3:
        raise KeyError(f'文件3缺少列: {missing3}')

    # ==============================
    # 4. 按时间把文件1拼接到文件2
    # ==============================
    print('[3/5] 按时间把文件1数据赋值到文件2...')

    if df1['时间'].duplicated().any():
        dup_times = df1.loc[df1['时间'].duplicated(keep=False), '时间'].sort_values()
        raise ValueError(f'文件1中存在重复时间戳，例如：\n{dup_times.head(20)}')

    df1_merge = df1.rename(columns={'时间': 'DateAndTime'})

    cols_to_merge = [
        'DateAndTime',
        '平均有功功率_风机导出_前10分钟均值',
        'ACTIVE_POWER_#56_对齐_前10分钟均值',
        'WINDSPEED_#56_对齐_前10分钟均值',
        'WINDDIRECTION_#56_对齐_前10分钟环形均值'
    ]

    # 仅保留文件1存在的时间戳
    df_out = df2.merge(df1_merge[cols_to_merge], on='DateAndTime', how='inner')
    unique_common_time_count = df_out.loc[
        df_out['DateAndTime'].isin(df1_merge['DateAndTime']),
        'DateAndTime'
    ].nunique()
    print(f'    文件1与文件2的交集时间点数: {unique_common_time_count:,}')

    # 仅对 Distance == 0 的行，把文件1中的风速/风向赋值给文件2中的 HWS(hub) / DIR(hub)
    mask_distance_zero = df_out['Distance'] == 0

    df_out.loc[mask_distance_zero, 'HWS(hub)'] = df_out.loc[
        mask_distance_zero, 'WINDSPEED_#56_对齐_前10分钟均值'
    ]
    df_out.loc[mask_distance_zero, 'DIR(hub)'] = df_out.loc[
        mask_distance_zero, 'WINDDIRECTION_#56_对齐_前10分钟环形均值'
    ]

    print(f'    已对 Distance == 0 的行回填 HWS(hub)/DIR(hub)，行数: {mask_distance_zero.sum():,}')

    # 删除'WINDSPEED_#56_对齐_前10分钟均值','WINDDIRECTION_#56_对齐_前10分钟环形均值'
    df_out = df_out.drop(columns=['WINDSPEED_#56_对齐_前10分钟均值', 'WINDDIRECTION_#56_对齐_前10分钟环形均值'])
    
    # ==============================
    # 5. 添加“风机故障”状态列
    # ==============================
    print('[4/5] 根据故障区间和10分钟窗口添加风机故障状态列...')

    df_out['风机故障'] = add_fault_status_for_10min_window(
        df=df_out,
        time_col='DateAndTime',
        fault_df=df3,
        start_col='start_time',
        end_col='end_time'
    )

    print(f"    风机故障=1 的行数: {(df_out['风机故障'] == 1).sum():,}")
    print(f"    风机故障=0 的行数: {(df_out['风机故障'] == 0).sum():,}")

    # ==============================
    # 6. 输出文件
    # ==============================
    print('[5/5] 保存输出文件...')

    df_out = df_out.sort_values(['DateAndTime', 'Distance']).reset_index(drop=True)
    df_out['DateAndTime'] = df_out['DateAndTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df_out.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f'完成！输出文件已保存到：\n{output_file}')


if __name__ == '__main__':
    main()