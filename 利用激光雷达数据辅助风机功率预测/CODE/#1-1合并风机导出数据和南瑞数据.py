import pandas as pd

def merge_wind_turbine_data(file_path1, file_path2, output_path='merged_output.csv'):
    """
    合并两个风机相关的CSV文件数据，基于时间戳对齐，并保存为新CSV文件。

    参数:
        file_path1 (str): 第一个CSV文件路径，应包含 'timestamp', 'ACTIVE_POWER_#56', 'WINDSPEED_#56' 
        'WINDDIRECTION_#56'列。
        file_path2 (str): 第二个CSV文件路径，应包含 '时间', '平均有功功率_风机导出' 列。
        output_path (str): 合并后数据保存的路径（默认为 'merged_output.csv'）

    返回:
        pd.DataFrame: 合并后的数据框
    """

    # 读取第一个文件并选择相关列
    df1 = pd.read_csv(file_path1, usecols=['timestamp', 'ACTIVE_POWER_#56', 'WINDSPEED_#56', 'WINDDIRECTION_#56'], encoding='gbk')
    df1.rename(columns={'timestamp': '时间'}, inplace=True)
    df1['时间'] = pd.to_datetime(df1['时间'])

    # 读取第二个文件并选择相关列
    df2 = pd.read_csv(file_path2, usecols=['时间', '平均有功功率_风机导出'],encoding='gbk')
    df2['时间'] = pd.to_datetime(df2['时间'])

    # 合并两个数据集
    merged_df = pd.merge(df1, df2, on='时间', how='inner')

    # 保存到新CSV文件
    merged_df.to_csv(output_path, index=False, encoding='gbk')

    return merged_df

if __name__ == "__main__":
    file_path1 = 'RAW_DATA\峡沙#56号风机20240830-20241121_南瑞.csv'
    file_path2 = 'RAW_DATA\#56风机功率_2024.csv'
    output_path = 'PROCESS_DATA\#1-1合并风机导出数据和南瑞数据.csv'
    
    merged_data = merge_wind_turbine_data(file_path1, file_path2, output_path)
    print("合并后的数据已保存到:", output_path)