#!/usr/bin/env python3
"""
ASVspoof 2019 平台真實用戶評估：精簡版 (只測試 upload + analyze)
包含完整時間統計 + 真假分數分析 + EER/AUC/P95指標
"""

import os
import time
import requests
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.metrics import roc_curve, roc_auc_score
warnings.filterwarnings('ignore')

# 配置 - 已更新為你的實際路徑
BASE_URL = "http://localhost:5001"  # 你的 Flask 服務
FLAC_DIR = Path("/home/carmen/asvspoof/datasets/LA/ASVspoof2019_LA_eval/flac")
DATASET_CSV = "/home/carmen/asvspoof/preprocess/eval_20_dataset.csv" 
RESULTS_CSV = "/home/carmen/asvspoof/results/platform_eval_results_20.csv"
SUMMARY_JSON = "/home/carmen/asvspoof/results/platform_eval_summary_20.json"

class PlatformEvaluator:
    def __init__(self):
        self.results = []
        self.real_scores = []
        self.fake_scores = []
        
    def load_dataset_from_csv(self, csv_path):
        """從CSV載入測試資料集"""
        df = pd.read_csv(csv_path)
        test_files = []
        for _, row in df.iterrows():
            filepath = FLAC_DIR / row['filename']
            if filepath.exists():
                # real→bonafide, fake→spoof
                label = 'bonafide' if row['label'] == 'real' else 'spoof'
                test_files.append((str(filepath), row['filename'], label))
            else:
                print(f"⚠️ 檔案不存在: {row['filename']}")
        print(f"✅ 載入CSV資料集: {len(test_files)} 個有效檔案")
        print(f"   真實: {sum(1 for _,_,l in test_files if l=='bonafide')}")
        print(f"   假: {sum(1 for _,_,l in test_files if l=='spoof')}")
        return test_files

    def test_upload_analyze(self, filepath, filename, label):
        """測試單一檔案：upload → analyze 全流程"""
        timings = {}
        try:
            # 1. 上傳計時
            start = time.time()
            with open(filepath, 'rb') as f:
                files = {'file': f}
                upload_resp = requests.post(f"{BASE_URL}/upload", files=files, timeout=30)
            timings['upload_time'] = time.time() - start

            if not upload_resp.json().get('success'):
                print(f"⚠️ 上傳失敗 {filename}: {upload_resp.text[:50]}")
                return None

            server_filename = upload_resp.json()['filename']

            # 2. 分析計時
            start = time.time()
            analyze_resp = requests.post(f"{BASE_URL}/analyze",
                                       json={'filename': server_filename}, 
                                       timeout=60)
            timings['analyze_time'] = time.time() - start

            if analyze_resp.json().get('success'):
                result = json.loads(analyze_resp.json()['result'])
                prob_real = result.get('prob_real', 0)
                prob_fake = result.get('prob_fake', 0)

                return {
                    'filename': filename,
                    'label': label,
                    'test_type': 'upload',
                    'prob_real': float(prob_real),
                    'prob_fake': float(prob_fake),
                    'upload_time': float(timings['upload_time']),
                    'analyze_time': float(timings['analyze_time']),
                    'total_time': float(timings['upload_time'] + timings['analyze_time']),
                    'size_mb': os.path.getsize(filepath) / (1024*1024)
                }
        except Exception as e:
            print(f"❌ 測試失敗 {filename}: {str(e)[:80]}")
        return None
    
    def run_evaluation(self, max_files=20):
        """執行完整評估 - 只測試 upload + analyze"""
        print("🔥 開始平台真實用戶評估 (upload + analyze only)...")
        
        # 1. 載入測試檔案
        test_files = self.load_dataset_from_csv(DATASET_CSV)
        test_files = test_files[:max_files]
        
        print(f"📊 準備測試 {len(test_files)} 個檔案")

        # 2. 只執行 Upload/Analyze 測試
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # 只測試 upload + analyze
            for filepath, filename, label in tqdm(test_files, desc="Upload/Analyze"):
                future = executor.submit(self.test_upload_analyze, filepath, filename, label)
                futures.append(('upload', future))

        # 3. 收集結果
        for test_type, future in tqdm(futures, desc="收集結果"):
            result = future.result()
            if result:
                self.results.append(result)
                if result['label'] == 'bonafide':
                    self.real_scores.append(result['prob_real'])
                else:
                    self.fake_scores.append(result['prob_real'])
        
        print(f"✅ 完成！成功測試 {len(self.results)} 個檔案")

    def generate_report(self):
        """生成完整報告 - 時間統計 + 真假分數 + EER/AUC/P95"""
        df = pd.DataFrame(self.results)
        if df.empty:
            print("❌ 無測試結果 - 請檢查 Flask 服務 http://localhost:5001 是否運行")
            print("💡 終端1: cd ~/asvspoof/program && python app.py")
            return

        # 🔧 確保時間欄位正確
        df['upload_time'] = df.get('upload_time', 0.0)
        df['analyze_time'] = df.get('analyze_time', 0.0)
        df['total_time'] = df['upload_time'] + df['analyze_time']

        # 分離真假聲音 (使用 Real score)
        real_df = df[df['label'] == 'bonafide']
        fake_df = df[df['label'] == 'spoof']
        
        real_scores = real_df['prob_real'].values if len(real_df) > 0 else np.array([])
        fake_scores = fake_df['prob_real'].values if len(fake_df) > 0 else np.array([])

        # 1️⃣ 時間統計
        time_stats = {
            'total_runtime': float(df['total_time'].sum()),      # 總運行時間
            'min_time': float(df['total_time'].min()),
            'max_time': float(df['total_time'].max()), 
            'avg_time': float(df['total_time'].mean()),
            'p95_time': float(df['total_time'].quantile(0.95)),
            'upload_avg': float(df['upload_time'].mean()),
            'analyze_avg': float(df['analyze_time'].mean()),
            'files_count': len(df)
        }

        # 2️⃣ 真實聲音統計 (Real score)
        real_stats = {
            'count': int(len(real_scores)),
            'min': float(np.min(real_scores)) if len(real_scores)>0 else 0.0,
            'max': float(np.max(real_scores)) if len(real_scores)>0 else 0.0,
            'avg': float(np.mean(real_scores)) if len(real_scores)>0 else 0.0
        }

        # 3️⃣ 虛假聲音統計 (Real score)  
        fake_stats = {
            'count': int(len(fake_scores)),
            'min': float(np.min(fake_scores)) if len(fake_scores)>0 else 0.0,
            'max': float(np.max(fake_scores)) if len(fake_scores)>0 else 0.0,
            'avg': float(np.mean(fake_scores)) if len(fake_scores)>0 else 0.0
        }

        # 4️⃣ 標準指標 (EER, AUC)
        def compute_eer(real_scores, fake_scores):
            if len(fake_scores) == 0:
                return 1.0, 0.5
            labels = [0]*len(real_scores) + [1]*len(fake_scores)
            scores = np.array(real_scores + fake_scores)
            fpr, tpr, thresholds = roc_curve(labels, 1-scores)  # fake_score = 1-real_score
            eer_idx = np.argmin(np.abs(fpr - tpr))
            return fpr[eer_idx], thresholds[eer_idx]

        eer, eer_threshold = compute_eer(real_scores.tolist(), fake_scores.tolist())
        auc_score = roc_auc_score([0]*len(real_scores) + [1]*len(fake_scores), 
                                real_scores.tolist() + fake_scores.tolist()) if len(fake_scores)>0 else 0.5

        # 5️⃣ 完整報告
        summary = {
            'dataset': {'total': len(df), 'real': real_stats['count'], 'fake': fake_stats['count']},
            'timing': time_stats,
            'real_scores': real_stats,
            'fake_scores': fake_stats,
            'metrics': {'eer': float(eer), 'auc': float(auc_score), 'eer_threshold': float(eer_threshold)}
        }

        # 保存CSV + JSON
        df.to_csv(RESULTS_CSV, index=False)
        with open(SUMMARY_JSON, 'w') as f:
            json.dump(summary, f, indent=2)

        # 📊 美觀報告輸出
        print("\n" + "="*90)
        print("🎯 PLATFORM COMPREHENSIVE EVALUATION REPORT")
        print("="*90)
        print(f"📁 總測試檔案: {summary['dataset']['total']:,}")
        print(f"🎵 真實:{summary['dataset']['real']:>4,} | 假:{summary['dataset']['fake']:>4,}")
        print()

        print("⏱️  TIME ANALYSIS (upload + analyze only)")
        print("-" * 60)
        print(f"💾 總運行時間:     {time_stats['total_runtime']:7.1f}s")
        print(f"⚡ 最快單檔:       {time_stats['min_time']:7.3f}s") 
        print(f"🐌 最慢單檔:       {time_stats['max_time']:7.3f}s")
        print(f"📈 平均單檔時間:   {time_stats['avg_time']:7.3f}s")
        print(f"🎯 P95時間:        {time_stats['p95_time']:7.3f}s")
        print(f"⬆️  上傳平均時間:  {time_stats['upload_avg']:7.3f}s")
        print(f"🔍 分析平均時間:   {time_stats['analyze_avg']:7.3f}s")
        print()

        print("🤖 MODEL PERFORMANCE (Real Probability Score)")
        print("-" * 60)
        print(f"✅ 真實聲音 (Real Score):")
        print(f"   最低: {real_stats['min']:7.4f} | 最高: {real_stats['max']:7.4f} | 平均: {real_stats['avg']:7.4f}")
        print(f"❌ 虛假聲音 (Real Score):")
        print(f"   最低: {fake_stats['min']:7.4f} | 最高: {fake_stats['max']:7.4f} | 平均: {fake_stats['avg']:7.4f}")
        print()

        print("📊 STANDARD DETECTION METRICS")
        print("-" * 60)
        print(f"🎯 EER (等錯誤率):     {summary['metrics']['eer']:.4f}")
        print(f"📈 AUC (ROC曲線):      {summary['metrics']['auc']:.4f}")
        print(f"⚖️  EER最佳閾值:      {summary['metrics']['eer_threshold']:.4f}")

        print(f"\n💾 報告已保存: {RESULTS_CSV} | {SUMMARY_JSON}")
        return summary

if __name__ == "__main__":
    evaluator = PlatformEvaluator()
    
    print(f"📁 使用 FLAC 路徑: {FLAC_DIR}")
    print(f"📁 使用 CSV 資料集: {DATASET_CSV}")
    print(f"🚀 預設測試 {min(20, len(pd.read_csv(DATASET_CSV)))} 個檔案")
    
    evaluator.run_evaluation(max_files=20)  # 只測試20個檔案
    summary = evaluator.generate_report()

