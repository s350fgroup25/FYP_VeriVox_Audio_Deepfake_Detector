#!/usr/bin/env python3
"""
ASVspoof 2021 DF 平台真實用戶評估：完整版 (修復generate_report缺失)
支援只有fake數據情況 + 完整時間統計 + EER/AUC計算
"""

import os
import time
import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
import queue
import warnings
warnings.filterwarnings('ignore')

# 🔄 2021 DF dataset配置
BASE_URL = "http://localhost:5001"  # 你的 Flask 服務
FLAC_DIR = Path("/home/carmen/asvspoof/asvspoof2021")
DATASET_CSV = "/home/carmen/asvspoof/preprocess/df_my_actual_1000_mapping_simple.csv"
RESULTS_CSV = "/home/carmen/asvspoof/results/platform_eval_results_df_1000_N1.csv"
SUMMARY_JSON = "/home/carmen/asvspoof/results/platform_eval_summary_df_1000_N1.json"

class PlatformEvaluator:
    def __init__(self):
        self.results = []
        self.real_scores = []
        self.fake_scores = []
        self.program_start_time = 0.0
        self.program_total_time = 0.0

    def load_dataset_from_csv(self, csv_path):
        """從CSV載入2021 DF測試資料集"""
        df = pd.read_csv(csv_path)
        test_files = []
        for _, row in df.iterrows():
            filepath = FLAC_DIR / row['filename']
            if filepath.exists():
                label = 'bonafide' if row['label'] == 'real' else 'spoof'
                test_files.append((str(filepath), row['filename'], label))
            else:
                print(f"⚠️ 檔案不存在: {row['filename']}")
        
        print(f"✅ 載入2021 DF CSV資料集: {len(test_files)} 個有效檔案")
        print(f"   真實(bonafide): {sum(1 for _,_,l in test_files if l=='bonafide')}")
        print(f"   假(spoof): {sum(1 for _,_,l in test_files if l=='spoof')}")
        return test_files

    def test_upload_analyze(self, filepath, filename, label):
        """測試單一檔案：upload → analyze 全流程"""
        timings = {}
        try:
            # 1. 上傳計時
            start = time.time()
            with open(filepath, 'rb') as f:
                files = {'file': f}
                upload_resp = requests.post(f"{BASE_URL}/upload", files=files, timeout=60)
            timings['upload_time'] = float(time.time() - start)

            if not upload_resp.ok or not upload_resp.json().get('success'):
                print(f"⚠️ 上傳失敗 {filename}")
                return None

            server_filename = upload_resp.json()['filename']

            # 2. 分析計時
            start = time.time()
            analyze_resp = requests.post(f"{BASE_URL}/analyze",
                                       json={'filename': server_filename},
                                       timeout=120)
            timings['analyze_time'] = float(time.time() - start)

            if analyze_resp.ok and analyze_resp.json().get('success'):
                result = json.loads(analyze_resp.json()['result'])
                prob_real = float(result.get('prob_real', 0))
                prob_fake = float(result.get('prob_fake', 0))

                return {
                    'filename': filename,
                    'label': label,
                    'test_type': 'upload',
                    'prob_real': prob_real,
                    'prob_fake': prob_fake,
                    'upload_time': timings['upload_time'],
                    'analyze_time': timings['analyze_time'],
                    'total_time': timings['upload_time'] + timings['analyze_time'],
                    'size_mb': float(os.path.getsize(filepath) / (1024*1024))
                }
        except Exception as e:
            print(f"❌ 測試失敗 {filename}: {str(e)[:80]}")
        return None

    def run_evaluation(self, max_files=1000, max_concurrent=1):
        """動態並發：analyze response回來就放新檔案"""
        print(f"🔥 開始2021 DF平台真實用戶評估 (最大{max_concurrent}並發)...")

        program_start = time.perf_counter()
        self.program_start_time = program_start

        test_files = self.load_dataset_from_csv(DATASET_CSV)
        test_files = test_files[:max_files]
        print(f"📊 準備測試 {len(test_files)} 個檔案，最大並發: {max_concurrent}")

        file_queue = queue.Queue()
        for filepath, filename, label in test_files:
            file_queue.put((filepath, filename, label))

        active_tasks = 0
        results_processed = 0

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []
            pbar = tqdm(total=len(test_files), desc="處理2021 DF檔案")

            while results_processed < len(test_files):
                if active_tasks < max_concurrent and not file_queue.empty():
                    try:
                        filepath, filename, label = file_queue.get_nowait()
                        future = executor.submit(self.test_upload_analyze, filepath, filename, label)
                        futures.append(future)
                        active_tasks += 1
                        pbar.set_postfix({'並發': active_tasks, '已完成': results_processed})
                    except queue.Empty:
                        pass

                done_futures = []
                for future in futures[:]:
                    if future.done():
                        result = future.result()
                        if result:
                            self.results.append(result)
                            if result['label'] == 'bonafide':
                                self.real_scores.append(result['prob_real'])
                            else:
                                self.fake_scores.append(result['prob_real'])
                        results_processed += 1
                        active_tasks -= 1
                        done_futures.append(future)
                        pbar.update(1)
                        pbar.set_postfix({'並發': active_tasks, '已完成': results_processed})

                for future in done_futures:
                    futures.remove(future)

                time.sleep(0.1)

            pbar.close()

        print(f"\n✅ 2021 DF測試完成！成功測試 {len(self.results)} 個檔案")
        program_end = time.perf_counter()
        self.program_total_time = program_end - program_start
        print(f"⏱️  程式整體運行時間: {self.program_total_time:.1f}s")

    def to_json_serializable(self, obj):
        """轉換為JSON可序列化格式"""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def generate_report(self):
        """🔥 完整報告生成 - 支援只有fake數據情況"""
        df = pd.DataFrame(self.results)
        if df.empty:
            print("❌ 無測試結果 - 請檢查 Flask 服務")
            return

        # 確保時間欄位正確
        df['upload_time'] = pd.to_numeric(df.get('upload_time', 0.0), errors='coerce').fillna(0.0)
        df['analyze_time'] = pd.to_numeric(df.get('analyze_time', 0.0), errors='coerce').fillna(0.0)
        df['total_time'] = df['upload_time'] + df['analyze_time']

        real_df = df[df['label'] == 'bonafide']
        fake_df = df[df['label'] == 'spoof']

        real_scores = real_df['prob_real'].values if len(real_df) > 0 else np.array([])
        fake_scores = fake_df['prob_real'].values if len(fake_df) > 0 else np.array([])

        # 🔥 完整時間統計
        time_stats = {
            'program_total': float(self.program_total_time),
            'total_runtime': float(df['total_time'].sum()),
            'min_time': float(df['total_time'].min()),
            'max_time': float(df['total_time'].max()),
            'avg_time': float(df['total_time'].mean()),
            'p95_time': float(df['total_time'].quantile(0.95)),
            'upload_avg': float(df['upload_time'].mean()),
            'analyze_avg': float(df['analyze_time'].mean()),
            'analyze_min': float(df['analyze_time'].min()),
            'analyze_max': float(df['analyze_time'].max()),
            'files_count': int(len(df))
        }

        # 🔥 分數統計
        real_stats = {
            'count': int(len(real_scores)),
            'min': float(np.min(real_scores)) if len(real_scores)>0 else 0.0,
            'max': float(np.max(real_scores)) if len(real_scores)>0 else 0.0,
            'avg': float(np.mean(real_scores)) if len(real_scores)>0 else 0.0
        }
        fake_stats = {
            'count': int(len(fake_scores)),
            'min': float(np.min(fake_scores)) if len(fake_scores)>0 else 0.0,
            'max': float(np.max(fake_scores)) if len(fake_scores)>0 else 0.0,
            'avg': float(np.mean(fake_scores)) if len(fake_scores)>0 else 0.0
        }

        # 🔥 EER計算 (支援只有fake情況)
        def compute_eer(real_scores, fake_scores):
            if len(real_scores) == 0 or len(fake_scores) == 0:
                return 1.0, 0.5, "缺少real或fake數據"
            labels = [0]*len(real_scores) + [1]*len(fake_scores)
            scores = np.array(real_scores + fake_scores)
            fpr, tpr, thresholds = roc_curve(labels, 1-scores)
            eer_idx = np.argmin(np.abs(fpr - tpr))
            return float(fpr[eer_idx]), float(thresholds[eer_idx]), None

        eer, eer_threshold, eer_note = compute_eer(real_scores.tolist(), fake_scores.tolist())
        
        auc_score = (roc_auc_score([0]*len(real_scores) + [1]*len(fake_scores),
                                  real_scores.tolist() + fake_scores.tolist()) 
                     if len(real_scores)>0 and len(fake_scores)>0 else None)

        # 🔥 精準時間統計輸出
        print("\n📊 總檔案數:   {:>3}".format(time_stats['files_count']))
        print("⏱️  程式整體運行時間: {:>6.1f}s".format(time_stats['program_total']))
        print("💾  各檔案總和時間: {:>6.1f}s".format(time_stats['total_runtime']))
        print("⚡ 最快單檔:      {:6.3f}s (upload+analyze)".format(time_stats['min_time']))
        print("🐌 最慢單檔:      {:6.3f}s (upload+analyze)".format(time_stats['max_time']))
        print("📈 平均單檔:      {:6.3f}s".format(time_stats['avg_time']))
        print("🎯 P95時間:       {:6.3f}s".format(time_stats['p95_time']))
        print()
        print("🔍 細分時間:")
        print("   📤 平均上傳:   {:6.3f}s".format(time_stats['upload_avg']))
        print("   ⚙️  平均分析:   {:6.3f}s".format(time_stats['analyze_avg']))
        print("   ⏱️  分析最快:   {:6.3f}s".format(time_stats['analyze_min']))
        print("   ⏱️  分析最慢:   {:6.3f}s".format(time_stats['analyze_max']))
        print()

        # 🔥 美觀報告
        print("\n" + "="*90)
        print("🎯 PLATFORM COMPREHENSIVE EVALUATION REPORT - ASVspoof2021 DF")
        print("="*90)
        print(f"📁 總測試檔案: {time_stats['files_count']}")
        print(f"🎵 真實(bonafide):{real_stats['count']:>4} | 假(spoof):{fake_stats['count']:>4}")
        print()
        print("⏱️  TIME ANALYSIS (upload + analyze)")
        print("-" * 50)
        print(f"💾 總運行時間:   {time_stats['total_runtime']:6.1f}s")
        print(f"⚡ 最快單檔:     {time_stats['min_time']:6.3f}s")
        print(f"🐌 最慢單檔:     {time_stats['max_time']:6.3f}s")
        print(f"📈 平均單檔:     {time_stats['avg_time']:6.3f}s")
        print(f"🎯 P95時間:      {time_stats['p95_time']:6.3f}s")
        print()
        print("🤖 MODEL PERFORMANCE (Real Score)")
        print("-" * 50)
        if real_stats['count'] > 0:
            print(f"✅ 真實: {real_stats['min']:6.4f} ~ {real_stats['max']:6.4f} (avg:{real_stats['avg']:6.4f})")
        print(f"❌ 假:  {fake_stats['min']:6.4f} ~ {fake_stats['max']:6.4f} (avg:{fake_stats['avg']:6.4f})")
        print()
        print("📊 METRICS")
        print("-" * 50)
        if eer_note:
            print(f"EER: N/A ({eer_note}) | AUC: N/A")
        else:
            print(f"EER: {eer:.4f} | AUC: {auc_score:.4f}")
        print("\n💾 報告完成!")

        # JSON安全報告
        summary = {
            'dataset': {'total': int(len(df)), 'real': real_stats['count'], 'fake': fake_stats['count']},
            'timing': time_stats,
            'real_scores': real_stats,
            'fake_scores': fake_stats,
            'metrics': {
                'eer': eer if not eer_note else None,
                'auc': float(auc_score) if auc_score else None,
                'eer_threshold': eer_threshold if not eer_note else None,
                'note': eer_note
            }
        }

        # 保存檔案
        df.to_csv(RESULTS_CSV, index=False)
        with open(SUMMARY_JSON, 'w') as f:
            json.dump(summary, f, indent=2, default=self.to_json_serializable)

        print(f"✅ CSV已保存: {RESULTS_CSV}")
        print(f"✅ JSON已保存: {SUMMARY_JSON}")

if __name__ == "__main__":
    evaluator = PlatformEvaluator()

    print("🚀 開始2021 DF評估流程...")
    print(f"📁 FLAC路徑: {FLAC_DIR}")
    print(f"📁 CSV資料集: {DATASET_CSV}")
    print(f"📁 結果保存至: {RESULTS_CSV}")

    # 執行評估
    evaluator.run_evaluation(max_files=100, max_concurrent=1)
    evaluator.generate_report()

