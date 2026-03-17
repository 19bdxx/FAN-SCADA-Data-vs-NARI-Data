import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# 读取CSV文件
df = pd.read_csv('PROCESS_DATA\峡沙56号_时间戳对齐后数据.csv')  # 替换为你的CSV文件路径
df['时间'] = pd.to_datetime(df['时间'])

# 初始化Dash应用
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=df['时间'].min(),
        end_date=df['时间'].max(),
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(id='line-chart')
])

@app.callback(
    Output('line-chart', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    filtered_df = df[(df['时间'] >= start_date) & (df['时间'] <= end_date)]
    fig = px.line(filtered_df, x='时间', y=['ACTIVE_POWER_#56_原始', 'ACTIVE_POWER_#56_对齐', '平均有功功率_风机导出'],
                   labels={'value': 'Power', '时间': 'Time'},
                   title='Active Power and Average Power over Time')
    return fig

if __name__ == '__main__':
    app.run(debug=True)