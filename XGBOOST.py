# ===================================== 1. 导入所需库 =====================================
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable
import xgboost as xgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ===================================== 2. 读取训练和测试数据 =====================================
# 导入数据集
df = pd.read_excel('特征波段输入.xlsx')
header = df.columns.tolist()
print(header)

# 自变量 (前6列)
X = df.iloc[:, 0:6].values
# 因变量 (第7列，盐分含量)
y = df.iloc[:, 6].values

print(X.shape)  # 应显示 (80, 6)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ====== 修复：确保目标变量是二维数组 ======
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# ===================================== 3. 数据归一化 =====================================
m_in = MinMaxScaler()
vp_train = m_in.fit_transform(X_train)
vp_test = m_in.transform(X_test)

m_out = MinMaxScaler()
vt_train = m_out.fit_transform(y_train)
vt_test = m_out.transform(y_test)

# ===================================== 4. 构建并训练 XGBoost 回归模型 =====================================
model = xgb.XGBRegressor(
    n_estimators=15,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
# 注意：使用ravel()将二维目标转换为一维
model.fit(vp_train, vt_train.ravel())

# ===================================== 5. 模型预测与反归一化 =====================================
yhat = model.predict(vp_test).reshape(-1, 1)
predicted_data = m_out.inverse_transform(yhat)

# ===================================== 6. 定义评估函数 =====================================
def mape(y_true, y_pred):
    record = []
    for index in range(len(y_true)):
        temp_mape = np.abs((y_pred[index] - y_true[index]) / (y_true[index] + 1e-5))
        record.append(temp_mape)
    return np.mean(record) * 100

def evaluate_forecasts(Ytest, predicted_data, n_out):
    mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic = [], [], [], [], []
    table = PrettyTable(['测试集指标', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'])
    for i in range(n_out):
        actual = [float(row[i]) for row in Ytest]
        predicted = [float(row[i]) for row in predicted_data]
        mse = mean_squared_error(actual, predicted)
        rmse = sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        MApe = mape(actual, predicted)
        r2 = r2_score(actual, predicted)
        table.add_row([f'第{i + 1}步预测结果指标：', mse, rmse, mae, f"{MApe:.2f}%", f"{r2 * 100:.2f}%"])
        mse_dic.append(mse)
        rmse_dic.append(rmse)
        mae_dic.append(mae)
        mape_dic.append(MApe)
        r2_dic.append(r2)
    return mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table

# ===================================== 7. 输出评估结果 =====================================
mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table = evaluate_forecasts(y_test, predicted_data, 1)
print(table)

# ===================================== 8. 保存预测结果到 Excel =====================================
results_df = pd.DataFrame({
    '预测值': predicted_data.flatten(),
    '真实值': y_test.flatten()
})
results_df['绝对误差'] = np.abs(results_df['预测值'] - results_df['真实值'])
results_df['相对误差(%)'] = 100 * results_df['绝对误差'] / (results_df['真实值'] + 1e-5)

output_path = "0607-XGBoost-SSC结果20.xlsx"
results_df.to_excel(output_path, index=False)
print(f"\n✅ 预测结果已保存为 Excel 文件：{output_path}")

# ===================================== 9. 可视化预测结果 =====================================
from matplotlib import rcParams

rcParams.update({
    "font.family": 'serif',
    "font.size": 10,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
    'axes.unicode_minus': False
})

plt.ion()
plt.figure(figsize=(8, 2), dpi=300)
x = range(1, len(predicted_data) + 1)
plt.tick_params(labelsize=5)

plt.plot(x, predicted_data, linestyle="--", linewidth=0.8, label='Predict', marker="o", markersize=2)
plt.plot(x, y_test, linestyle="-", linewidth=0.5, label='Real', marker="x", markersize=2)
plt.legend(loc='upper right', frameon=False, fontsize=5)

plt.xlabel("Sample points", fontsize=5)
plt.ylabel("Value", fontsize=5)
plt.title(f"The prediction result (XGBoost):\nMAPE: {mape(y_test, predicted_data):.2f}%", fontsize=5)

plt.ioff()
plt.show()