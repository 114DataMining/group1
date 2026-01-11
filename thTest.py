import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 讀取你的結果檔
df = pd.read_csv('test_prediction_results_best_th.csv')

y_test = df['True_Label']
y_prob = df['Probability']

# 設定測試範圍：從 0.612 到 0.80，每隔 0.01 測一次
thresholds = np.arange(0.612, 0.80, 0.01)

print(f"{'Threshold':<10} | {'FP (誤判)':<8} | {'FN (漏判)':<8} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
print("-" * 75)

for th in thresholds:
    # 產生預測結果
    y_pred_th = (y_prob >= th).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_th).ravel()
    prec = precision_score(y_test, y_pred_th, zero_division=0)
    rec = recall_score(y_test, y_pred_th, zero_division=0)
    f1 = f1_score(y_test, y_pred_th, zero_division=0)
    
    print(f"{th:.4f}     | {fp:<8} | {fn:<8} | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")
    
    # 提醒：當 Recall 掉到 0.5 以下時
    if rec < 0.5:
        print(f"   *** 警告: Recall 已低於 0.5 ({rec:.4f}) ***")