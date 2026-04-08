#!/usr/bin/env python3
"""
🔥 正確版：只有fake數據的真實準確率計算
假檔案中 prob_real < 0.5 的比例 = Accuracy
"""

import pandas as pd
import numpy as np
import json
import os

# 🔄 DF結果檔案
RESULTS_CSV = "/home/carmen/asvspoof/results/platform_eval_results_df_1000_N1.csv"
SUMMARY_JSON = "/home/carmen/asvspoof/results/platform_eval_summary_df_1000_N1.json"

def analyze_only_fake_data(df):
    """🔥 只有偽造數據的真實分析"""
    fake_scores = df['prob_real'].values
    
    # 🔥 正確的準確率：fake中 prob_real < 0.5 的比例
    correct_fake = np.mean(fake_scores < 0.5)
    
    # 🔥 統計
    fake_stats = {
        'count': len(fake_scores),
        'min': float(np.min(fake_scores)),
        'max': float(np.max(fake_scores)),
        'avg': float(np.mean(fake_scores)),
        'correct_05': float(correct_fake),  # fake被正確識別的比例
        'wrong_05': float(np.mean(fake_scores >= 0.5))  # fake錯判為real的比例
    }
    
    print(f"\n📊 偽造檔案分析 (64個)")
    print(f"✅ 正確識別 (<0.5): {fake_stats['correct_05']:.1%} ({fake_stats['correct_05']*64:.0f}個)")
    print(f"❌ 錯判為real (≥0.5): {fake_stats['wrong_05']:.1%} ({fake_stats['wrong_05']*64:.0f}個)")
    print(f"⚠️ 最高fake分數: {fake_stats['max']:.4f}")
    
    return fake_stats

def main():
    print("🔥 DF只有偽造數據 - 正確分析...")
    
    # 載入
    df = pd.read_csv(RESULTS_CSV)
    print(f"📁 載入 {len(df)} 個檔案 (全偽造)")
    
    # 🔥 只有fake數據的真實分析
    fake_stats = analyze_only_fake_data(df)
    
    # 🔥 生成報告
    print("\n" + "="*60)
    print("🎯 DF偽造數據評估報告")
    print("="*60)
    print(f"📁 偽造檔案: {fake_stats['count']}")
    print(f"🎯 prob_real平均: {fake_stats['avg']:.4f}")
    print(f"✅ 檢測率@0.5: {fake_stats['correct_05']:.1%}")
    print(f"❌ 漏檢率@0.5: {fake_stats['wrong_05']:.1%}")
    
    # JSON報告
    summary = {
        'dataset': {
            'total': len(df),
            'real': 0,
            'fake': len(df),
            'note': '只有偽造數據測試'
        },
        'fake_analysis': fake_stats,
        'recommendation': {
            'threshold_05_accuracy': fake_stats['correct_05'],
            'leakage_rate': fake_stats['wrong_05'],
            'status': '模型有效但有少量漏檢'
        }
    }
    
    # 保存
    os.makedirs(os.path.dirname(SUMMARY_JSON), exist_ok=True)
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ 報告保存: {SUMMARY_JSON}")
    print("🎉 分析完成！")

if __name__ == "__main__":
    main()

