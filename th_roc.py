import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 1. 讀取資料
df = pd.read_csv('test_prediction_results_best_th.csv')

y_test = df['True_Label']
y_prob = df['Probability']

# 2. 計算 ROC 曲線數據
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 3. 定義輔助函式
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# --- 設定閾值 ---
th_default = 0.5
th_balanced = 0.612
th_final = 0.68  # 最終決定

idx_default = find_nearest_idx(thresholds, th_default)
idx_balanced = find_nearest_idx(thresholds, th_balanced)
idx_final = find_nearest_idx(thresholds, th_final)

# ==========================================
# ★ 計算混淆矩陣與進階指標
# ==========================================
y_pred_final = (y_prob >= th_final).astype(int)
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()

# 計算 Sensitivity (Recall) 與 Specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
combined_score = sensitivity + specificity

print(f"\n=== 最終選擇閾值 (Threshold = {th_final}) 詳細指標 ===")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity         : {specificity:.4f}")
print(f"Sum (Sens + Spec)   : {combined_score:.4f} (大於 1 代表優於隨機)")
print("=======================================================\n")

# 4. 開始畫圖
plt.figure(figsize=(10, 8))

# ROC 曲線
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

# 標記關鍵點
plt.scatter(fpr[idx_default], tpr[idx_default], s=100, c='gray', marker='o', label=f'Default (Th={th_default})', zorder=5)
plt.scatter(fpr[idx_balanced], tpr[idx_balanced], s=100, c='blue', marker='^', label=f'Balanced (Th={th_balanced})', zorder=5)
plt.scatter(fpr[idx_final], tpr[idx_final], s=200, c='red', marker='*', label=f'Clinical Optimal (Th={th_final})', zorder=10)
plt.plot([0, fpr[idx_final]], [1, tpr[idx_final]], 'r:', alpha=0.3)

# ==========================================
# ★ 更新圖表文字方塊 (包含 Sens + Spec)
# ==========================================
cm_text = (
    f"Threshold = {th_final}\n"
    f"--------------------\n"
    f"TP: {tp:<3} | FN: {fn:<3}\n"
    f"FP: {fp:<3} | TN: {tn:<3}\n"
    f"--------------------\n"
    f"Sens: {sensitivity:.3f}\n"
    f"Spec: {specificity:.3f}\n"
    f"Sum : {combined_score:.3f}"
)

# 放在圖的右下角 (位置可微調)
plt.text(0.6, 0.15, cm_text, fontsize=11, family='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="red", alpha=0.9))

# 5. 設定圖表細節
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)\n(Lower is Better)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)\n(Higher is Better)')
plt.title('ROC Analysis with Specificity & Sensitivity Check')
plt.legend(loc="lower right", bbox_to_anchor=(1, 0.45)) # 把圖例稍微往上移，避開文字框
plt.grid(True, alpha=0.3)

# 6. 存檔
plt.savefig('roc_curve_final_with_metrics.png', dpi=300)
plt.show()

print("圖表已生成，包含 Sens+Spec 指標。")