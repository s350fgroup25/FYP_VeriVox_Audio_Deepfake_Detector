#!/usr/bin/env python3
"""
從 platform_eval_results.csv 生成詳細評估報告
包含 EER計算、ROC曲線數據、分佈圖表等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 配置
RESULTS_CSV = "/home/carmen/asvspoof/program/platform_eval_results.csv"
REPORT_HTML = "/home/carmen/asvspoof/results/detailed_eval_report.html"
SUMMARY_JSON = "/home/carmen/asvspoof/results/detailed_eval_summary.json"

def load_results():
    """載入評估結果"""
    if not Path(RESULTS_CSV).exists():
        print(f"❌ 找不到 {RESULTS_CSV}")
        return None
    
    df = pd.read_csv(RESULTS_CSV)
    print(f"✅ 載入 {len(df)} 個測試結果")
    print(f"   真實樣本: {len(df[df['label']=='bonafide'])}")
    print(f"   假樣本: {len(df[df['label']=='spoof'])}")
    return df

def calculate_eer(scores_real, scores_fake):
    """計算 EER (Equal Error Rate)"""
    # 合併真假分數
    labels = [0] * len(scores_real) + [1] * len(scores_fake)  # 0=real, 1=fake
    scores = scores_real + scores_fake
    
    # ROC曲線
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer_threshold = thresholds[np.nanargmin(np.absolute((1-tpr) - fpr))]
    eer = np.min(np.absolute((1-tpr) - fpr))
    
    return eer, fpr, tpr, thresholds, eer_threshold

def generate_detailed_report(df):
    """生成詳細報告"""
    if df.empty:
        print("❌ 無數據")
        return
    
    # 🔢 基本統計
    real_scores = df[df['label']=='bonafide']['prob_real'].values
    fake_scores = df[df['label']=='spoof']['prob_real'].values
    
    # EER計算
    eer, fpr, tpr, thresholds, eer_threshold = calculate_eer(real_scores, fake_scores)
    auc_score_val = roc_auc_score([0]*len(real_scores) + [1]*len(fake_scores), 
                                list(real_scores) + list(fake_scores))
    
    # 時間統計
    avg_total_time = df['total_time'].mean()
    p95_time = df['total_time'].quantile(0.95)
    
    # 各頁面表現
    upload_df = df[df['test_type']=='upload']
    record_df = df[df['test_type']=='record']
    
    summary = {
        'total_tests': len(df),
        'real_count': len(real_scores),
        'fake_count': len(fake_scores),
        
        # 模型表現
        'eer': float(eer),
        'auc': float(auc_score_val),
        'eer_threshold': float(eer_threshold),
        
        # 分數統計
        'real_avg': float(np.mean(real_scores)),
        'real_std': float(np.std(real_scores)),
        'real_min': float(np.min(real_scores)),
        'real_max': float(np.max(real_scores)),
        
        'fake_avg': float(np.mean(fake_scores)),
        'fake_std': float(np.std(fake_scores)),
        'fake_min': float(np.min(fake_scores)),
        'fake_max': float(np.max(fake_scores)),
        
        # 時間統計
        'avg_total_time': float(avg_total_time),
        'p95_total_time': float(p95_time),
        'upload_avg_time': float(upload_df['total_time'].mean() if len(upload_df)>0 else 0),
        'record_avg_time': float(record_df['total_time'].mean() if len(record_df)>0 else 0),
    }
    
    # 保存JSON摘要
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("🎯 詳細評估報告")
    print("="*80)
    print(f"總測試數: {summary['total_tests']:,}")
    print(f"真實/假: {summary['real_count']:,} / {summary['fake_count']:,}")
    print()
    print("🤖 模型表現:")
    print(f"  EER: {summary['eer']:.4f} (閾值: {summary['eer_threshold']:.4f})")
    print(f"  AUC: {summary['auc']:.4f}")
    print()
    print("📊 分數統計:")
    print(f"  真實: {summary['real_avg']:.4f}±{summary['real_std']:.4f} [{summary['real_min']:.4f}~{summary['real_max']:.4f}]")
    print(f"  虛假: {summary['fake_avg']:.4f}±{summary['fake_std']:.4f} [{summary['fake_min']:.4f}~{summary['fake_max']:.4f}]")
    print()
    print("⏱️  性能統計:")
    print(f"  平均總時間: {summary['avg_total_time']:.3f}s (P95: {summary['p95_total_time']:.3f}s)")
    
    return summary

def create_visual_report(df, summary):
    """生成圖表報告"""
    real_scores = df[df['label']=='bonafide']['prob_real'].values
    fake_scores = df[df['label']=='spoof']['prob_real'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 分數分佈直方圖
    axes[0,0].hist(real_scores, bins=50, alpha=0.7, label='Real', color='green', density=True)
    axes[0,0].hist(fake_scores, bins=50, alpha=0.7, label='Fake', color='red', density=True)
    axes[0,0].axvline(summary['real_avg'], color='green', linestyle='--', label=f'Real Avg: {summary["real_avg"]:.3f}')
    axes[0,0].axvline(summary['fake_avg'], color='red', linestyle='--', label=f'Fake Avg: {summary["fake_avg"]:.3f}')
    axes[0,0].axvline(summary['eer_threshold'], color='blue', linestyle=':', label=f'EER阈值: {summary["eer_threshold"]:.3f}')
    axes[0,0].set_xlabel('Real Probability')
    axes[0,0].set_ylabel('密度')
    axes[0,0].set_title('分數分佈')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. ROC曲線
    labels = [0]*len(real_scores) + [1]*len(fake_scores)
    scores = list(real_scores) + list(fake_scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0,1].plot([0,1], [0,1], color='navy', lw=1, linestyle='--')
    axes[0,1].set_xlim([0.0, 1.0])
    axes[0,1].set_ylim([0.0, 1.05])
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC曲線')
    axes[0,1].legend(loc="lower right")
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 時間分佈
    axes[1,0].hist(df['total_time'], bins=50, alpha=0.7, color='blue')
    axes[1,0].axvline(summary['avg_total_time'], color='red', linestyle='--', label=f'Avg: {summary["avg_total_time"]:.3f}s')
    axes[1,0].axvline(summary['p95_total_time'], color='orange', linestyle=':', label=f'P95: {summary["p95_total_time"]:.3f}s')
    axes[1,0].set_xlabel('總時間 (秒)')
    axes[1,0].set_ylabel('頻次')
    axes[1,0].set_title('處理時間分佈')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 測試類型比較
    test_types = df['test_type'].value_counts()
    upload_times = df[df['test_type']=='upload']['total_time']
    record_times = df[df['test_type']=='record']['total_time']
    
    x = np.arange(len(test_types))
    width = 0.35
    axes[1,1].bar(x, test_types.values, width, label='測試次數', alpha=0.7)
    axes[1,1].set_xlabel('測試類型')
    axes[1,1].set_ylabel('數量')
    axes[1,1].set_title('測試類型統計')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(test_types.index)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('eval_visual_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 圖表已保存: eval_visual_report.png")

if __name__ == "__main__":
    # 載入數據
    df = load_results()
    if df is None:
        exit(1)
    
    # 生成報告
    summary = generate_detailed_report(df)
    
    # 生成圖表
    create_visual_report(df, summary)
    
    print(f"\n🎉 完整報告完成！")
    print(f"   📊 JSON摘要: {SUMMARY_JSON}")
    print(f"   🖼️  視覺化圖表: eval_visual_report.png")

