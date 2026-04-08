#!/usr/bin/env python3
"""
ASVspoof 修復工具：自動重試失敗檔案 + 更新CSV + JSON
完整更新 platform_eval_results_1000.csv + platform_eval_summary_1000.json
"""

import os
import pandas as pd
import requests
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# 🔄 配置路徑
BASE_URL = "http://localhost:5001"
FLAC_DIR = Path("/home/carmen/asvspoof/datasets/LA/ASVspoof2019_LA_eval/flac")
RESULTS_CSV = "/home/carmen/asvspoof/results/platform_eval_results_1000.csv"
SUMMARY_JSON = "/home/carmen/asvspoof/results/platform_eval_summary_1000.json"
CHECKPOINT_CSV = "/home/carmen/asvspoof/results/platform_eval_checkpoint.csv"
RETRY_CSV = "/home/carmen/asvspoof/results/retry_failed_files.csv"

class FailedFilesRetrier:
    def __init__(self):
        self.results = []
        self.retry_count = 0

    def find_failed_files(self):
        """🔍 找出只upload沒結果的檔案"""
        print("🔍 分析檔案狀態...")
        
        if not os.path.exists(CHECKPOINT_CSV) or not os.path.exists(RESULTS_CSV):
            print("❌ 找不到必要檔案")
            return []
        
        checkpoint_df = pd.read_csv(CHECKPOINT_CSV)
        results_df = pd.read_csv(RESULTS_CSV)
        
        checkpoint_files = set(checkpoint_df['filename'].tolist())
        results_files = set(results_df['filename'].tolist())
        
        # 🔥 只upload沒analyze結果的檔案
        failed_files = checkpoint_files - results_files
        print(f"✅ checkpoint總數: {len(checkpoint_files)}")
        print(f"✅ results總數: {len(results_files)}")
        print(f"🔥 發現 {len(failed_files)} 個只upload沒結果的檔案")
        
        return list(failed_files), checkpoint_df, results_df

    def test_single_file(self, filename):
        """重試單一檔案（完整計時）"""
        filepath = FLAC_DIR / filename
        if not filepath.exists():
            print(f"⚠️ 檔案不存在: {filename}")
            return None
        
        timings = {}
        try:
            print(f"🔄 重試 {filename}...")
            
            # 上傳計時
            start = time.time()
            with open(filepath, 'rb') as f:
                files = {'file': f}
                upload_resp = requests.post(f"{BASE_URL}/upload", files=files, timeout=60)
            timings['upload_time'] = float(time.time() - start)

            if not upload_resp.ok or not upload_resp.json().get('success'):
                print(f"❌ 上傳失敗 {filename}")
                return None

            server_filename = upload_resp.json()['filename']

            # 分析計時
            start = time.time()
            analyze_resp = requests.post(f"{BASE_URL}/analyze",
                                       json={'filename': server_filename},
                                       timeout=120)
            timings['analyze_time'] = float(time.time() - start)

            if analyze_resp.ok and analyze_resp.json().get('success'):
                result = json.loads(analyze_resp.json()['result'])
                prob_real = float(result.get('prob_real', 0))
                prob_fake = float(result.get('prob_fake', 0))
                
                self.retry_count += 1
                print(f"✅ 重試成功 #{self.retry_count}: {filename}")
                
                return {
                    'filename': filename,
                    'label': 'retry',
                    'test_type': 'upload',
                    'prob_real': prob_real,
                    'prob_fake': prob_fake,
                    'upload_time': timings['upload_time'],
                    'analyze_time': timings['analyze_time'],
                    'total_time': timings['upload_time'] + timings['analyze_time'],
                    'size_mb': float(os.path.getsize(filepath) / (1024*1024))
                }
            else:
                print(f"❌ 分析失敗 {filename}")
                
        except Exception as e:
            print(f"❌ 重試異常 {filename}: {str(e)[:50]}")
        return None

    def run_retry(self, max_concurrent=5):
        """執行重試"""
        failed_files, checkpoint_df, results_df = self.find_failed_files()
        if not failed_files:
            print("✅ 沒有需要重試的檔案")
            self.update_all_files()
            return
        
        print(f"📋 重試清單 ({len(failed_files)} 個檔案):")
        for i, filename in enumerate(failed_files[:5], 1):
            print(f"   {i}. {filename}")
        if len(failed_files) > 5:
            print(f"   ... 還有 {len(failed_files)-5} 個")
        
        # 多線程重試
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(self.test_single_file, filename) for filename in failed_files]
            for future in tqdm(futures, desc="重試進度"):
                result = future.result()
                if result:
                    self.results.append(result)
        
        # 保存重試結果
        if self.results:
            retry_df = pd.DataFrame(self.results)
            retry_df.to_csv(RETRY_CSV, index=False)
            print(f"\n✅ 重試完成！成功 {len(self.results)}/{len(failed_files)} 個")

    def update_all_files(self):
        """🔥 完整更新 CSV + JSON"""
        print("🔄 完整更新所有檔案...")
        
        if not os.path.exists(RESULTS_CSV):
            print("❌ 找不到結果檔案")
            return
        
        # ✅ 1. 更新 platform_eval_results_1000.csv
        df = pd.read_csv(RESULTS_CSV)
        print(f"📊 載入CSV: {len(df)} 個檔案")
        
        # 確保時間欄位正確
        df['upload_time'] = pd.to_numeric(df.get('upload_time', 0.0), errors='coerce').fillna(0.0)
        df['analyze_time'] = pd.to_numeric(df.get('analyze_time', 0.0), errors='coerce').fillna(0.0)
        df['total_time'] = df['upload_time'] + df['analyze_time']
        
        # 重新排序
        df = df.sort_values('filename').reset_index(drop=True)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"✅ CSV已更新: {len(df)} 個檔案")

        # ✅ 2. 更新 platform_eval_summary_1000.json
        real_df = df[df['label'] == 'bonafide']
        fake_df = df[df['label'].isin(['spoof', 'retry'])]
        
        real_scores = real_df['prob_real'].values if len(real_df) > 0 else np.array([])
        fake_scores = fake_df['prob_real'].values if len(fake_df) > 0 else np.array([])

        time_stats = {
            'total_runtime': float(df['total_time'].sum()),
            'avg_time': float(df['total_time'].mean()),
            'p95_time': float(df['total_time'].quantile(0.95)),
            'upload_avg': float(df['upload_time'].mean()),
            'analyze_avg': float(df['analyze_time'].mean()),
            'files_count': int(len(df))
        }

        real_stats = {
            'count': int(len(real_scores)),
            'avg': float(np.mean(real_scores)) if len(real_scores)>0 else 0.0
        }
        fake_stats = {
            'count': int(len(fake_scores)),
            'avg': float(np.mean(fake_scores)) if len(fake_scores)>0 else 0.0
        }

        def compute_eer(real_scores, fake_scores):
            if len(fake_scores) == 0:
                return 1.0, 0.5
            labels = [0]*len(real_scores) + [1]*len(fake_scores)
            scores = np.array(real_scores + fake_scores)
            fpr, tpr, thresholds = roc_curve(labels, 1-scores)
            eer_idx = np.argmin(np.abs(fpr - tpr))
            return float(fpr[eer_idx]), float(thresholds[eer_idx])

        eer, eer_threshold = compute_eer(real_scores.tolist(), fake_scores.tolist())
        auc_score = float(roc_auc_score([0]*len(real_scores) + [1]*len(fake_scores),
                                      real_scores.tolist() + fake_scores.tolist()) if len(fake_scores)>0 else 0.5)

        summary = {
            'dataset': {'total': int(len(df)), 'real': real_stats['count'], 'fake': fake_stats['count']},
            'timing': time_stats,
            'real_scores': real_stats,
            'fake_scores': fake_stats,
            'metrics': {'eer': eer, 'auc': auc_score, 'eer_threshold': eer_threshold}
        }

        with open(SUMMARY_JSON, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ JSON已更新: {len(df)} 個檔案")
        print(f"🎯 EER: {eer:.4f} | AUC: {auc_score:.4f}")

    def merge_results(self):
        """合併重試結果到主CSV"""
        if not os.path.exists(RETRY_CSV) or len(self.results) == 0:
            print("⚠️ 沒有重試結果可合併")
            self.update_all_files()
            return
        
        main_df = pd.read_csv(RESULTS_CSV)
        retry_df = pd.read_csv(RETRY_CSV)
        
        # 合併並去重
        combined_df = pd.concat([main_df, retry_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['filename'], inplace=True)
        combined_df = combined_df.sort_values('filename').reset_index(drop=True)
        
        combined_df.to_csv(RESULTS_CSV, index=False)
        print(f"✅ 主CSV已更新: {len(combined_df)} 個檔案 (新增 {len(retry_df)} 個)")
        
        # 🔥 自動更新所有檔案
        self.update_all_files()

def main():
    retrier = FailedFilesRetrier()
    
    print("="*70)
    print("🔥 ASVspoof 完整修復工具 (CSV + JSON)")
    print("="*70)
    
    # 執行重試
    retrier.run_retry(max_concurrent=5)
    
    # 合併並完整更新
    retrier.merge_results()
    
    print("\n🎉 全部完成！兩個檔案都已更新:")
    print(f"   ✅ {RESULTS_CSV}")
    print(f"   ✅ {SUMMARY_JSON}")
    print("\n現在狀態完美，可以繼續主程式：")
    print("python3 eval_platform.py")

if __name__ == "__main__":
    main()

