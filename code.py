import pandas as pd
import math

df = pd.read_excel('11.20-CDE.xlsx', header=None, engine='openpyxl')

x_values = df.iloc[0, :]  # 第 1 行的 X 值（对应点 B、C、D）
y_values = df.iloc[1:, :]  # 第 2 行到最后一行的 Y 值

def calculate_tangent_at_C(x_values, y_values_row):

    B = (x_values[0], y_values_row[0])
    C = (x_values[1], y_values_row[1])
    D = (x_values[2], y_values_row[2])

    slope_BC = (C[1] - B[1]) / (C[0] - B[0])
    slope_CD = (D[1] - C[1]) / (D[0] - C[0])

    tangent_C = (slope_CD - slope_BC) / (1 + slope_BC * slope_CD)

    return tangent_C

tangent_values = y_values.apply(lambda row: calculate_tangent_at_C(x_values, row), axis=1)
result_df = pd.DataFrame({"Tangent at C": tangent_values})
result_df.to_excel('tangent_values.xlsx', index=False)
print("计算完成，结果已保存为 'tangent_values.xlsx'")