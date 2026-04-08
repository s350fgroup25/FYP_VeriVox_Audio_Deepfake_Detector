#!/usr/bin/env python3
"""
ASVspoof 2019 平台真實用戶評估：完美斷點續傳 + 智能合併等待
每檔2s冷卻 + 每100檔1分鐘休息（analyze_time純處理時間）
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

# 🔄 配置路徑
BASE_URL = "http://localhost:5001"
FLAC_DIR = Path("/home/carmen/asvspoof/datasets/LA/ASVspoof2019_LA_eval/flac")
DATASET_CSV = "/home/carmen/asvspoof/preprocess/eval_100_dataset.csv"
RESULTS_CSV = "/home/carmen/asvspoof/results/platform_eval_results_N100.csv"
SUMMARY_JSON = "/home/carmen/asvspoof/results/platform_eval_summary_N100.json"
CHECKPOINT_CSV = "/home/carmen/asvspoof/results/platform_eval_checkpoint100.csv"

class PlatformEvaluator:
    def __init__(self):
        self.results = []
        self.real_scores = []
        self.fake_scores = []
        self.program_start_time = 0.0
        self.program_total_time = 0.0

    def load_checkpoint(self):
        """正確載入所有已完成檔案"""
        if os.path.exists(CHECKPOINT_CSV):
            try:
                df = pd.read_csv(CHECKPOINT_CSV)
                completed = set(df['filename'].tolist())
                print(f"✅ 載入斷點: {len(completed)} 個檔案")
                return completed
            except Exception as e:
                print(f"⚠️ 斷點檔案損壞: {e}")
        return set()

    def save_checkpoint(self, filename):
        """累積保存所有已完成檔案（不覆蓋）"""
        if not os.path.exists(CHECKPOINT_CSV):
            checkpoint_data = []
        else:
            try:
                checkpoint_df = pd.read_csv(CHECKPOINT_CSV)
                checkpoint_data = checkpoint_df['filename'].tolist()
            except:
                checkpoint_data = []
        
        if filename not in checkpoint_data:
            checkpoint_data.append(filename)
            pd.DataFrame({'filename': checkpoint_data}).to_csv(CHECKPOINT_CSV, index=False)

    def load_dataset_from_csv(self, csv_path, skip_completed=set()):
        """載入資料集，跳過已完成的檔案"""
        df = pd.read_csv(csv_path)
        test_files = []
        skipped_count = 0
        for _, row in df.iterrows():
            if row['filename'] not in skip_completed:
                filepath = FLAC_DIR / row['filename']
                if filepath.exists():
                    label = 'bonafide' if row['label'] == 'real' else 'spoof'
                    test_files.append((str(filepath), row['filename'], label))
            else:
                skipped_count += 1
        
        print(f"✅ 本輪新檔案: {len(test_files)} 個（跳過 {skipped_count} 已完成）")
        print(f"   真實: {sum(1 for _,_,l in test_files if l=='bonafide')}")
        print(f"   假: {sum(1 for _,_,l in test_files if l=='spoof')}")
        return test_files

    def test_upload_analyze(self, filepath, filename, label):
        """測試單一檔案：純處理時間（不含等待）"""
        timings = {}
        try:
            # 上傳計時
            start = time.time()
            with open(filepath, 'rb') as f:
                files = {'file': f}
                upload_resp = requests.post(f"{BASE_URL}/upload", files=files, timeout=60)
            timings['upload_time'] = float(time.time() - start)

            if not upload_resp.ok or not upload_resp.json().get('success'):
                print(f"⚠️ 上傳失敗 {filename}")
                return None

            server_filename = upload_resp.json()['filename']

            # 分析計時（純處理時間）
            start = time.time()
            analyze_resp = requests.post(f"{BASE_URL}/analyze",
                                       json={'filename': server_filename},
                                       timeout=120)
            timings['analyze_time'] = float(time.time() - start)

            if analyze_resp.ok and analyze_resp.json().get('success'):
                result = json.loads(analyze_resp.json()['result'])
                prob_real = float(result.get('prob_real', 0))
                prob_fake = float(result.get('prob_fake', 0))

                result_data = {
                    'filename': filename,
                    'label': label,
                    'test_type': 'upload',
                    'prob_real': prob_real,
                    'prob_fake': prob_fake,
                    'upload_time': timings['upload_time'],
                    'analyze_time': timings['analyze_time'],  # ✅ 純處理時間
                    'total_time': timings['upload_time'] + timings['analyze_time'],
                    'size_mb': float(os.path.getsize(filepath) / (1024*1024))
                }
                
                self.save_checkpoint(filename)
                return result_data
                
        except Exception as e:
            print(f"❌ 測試失敗 {filename}: {str(e)[:80]}")
        return None

    def run_evaluation(self, max_files=1000, max_concurrent=10, cooldown_sec=2):
        """智能合併等待：每檔2s + 每100檔1分鐘"""
        print(f"🔥 開始評估 (並發={max_concurrent}, 每檔{cooldown_sec}s+每100檔1分鐘)...")

        program_start = time.perf_counter()
        self.program_start_time = program_start

        # 載入斷點
        completed_filenames = self.load_checkpoint()
        print(f"📊 歷史總完成: {len(completed_filenames)} 個檔案")
        
        test_files = self.load_dataset_from_csv(DATASET_CSV, completed_filenames)
        test_files = test_files[:max_files]
        
        if not test_files:
            print("✅ 所有檔案已完成！")
            return

        # 載入歷史結果
        if os.path.exists(RESULTS_CSV):
            history_df = pd.read_csv(RESULTS_CSV)
            self.results = history_df.to_dict('records')
            print(f"📥 載入歷史結果: {len(self.results)} 個")
        else:
            self.results = []

        file_queue = queue.Queue()
        for filepath, filename, label in test_files:
            file_queue.put((filepath, filename, label))

        active_tasks = 0
        results_processed = 0
        batch_count = 0
        
        try:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = []
                pbar = tqdm(total=len(test_files), desc="本輪進度")
                
                while results_processed < len(test_files):
                    # 送新任務
                    if active_tasks < max_concurrent and not file_queue.empty():
                        try:
                            filepath, filename, label = file_queue.get_nowait()
                            future = executor.submit(self.test_upload_analyze, filepath, filename, label)
                            futures.append(future)
                            active_tasks += 1
                        except queue.Empty:
                            pass
                    
                    # 合併處理完成任務
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
                            batch_count += 1
                            active_tasks -= 1
                            done_futures.append(future)
                            
                            # 🔥 智能合併等待
                            total_wait = cooldown_sec
                            if batch_count % 100 == 0:
                                total_wait += 58  # 2+58=60秒
                                print(f"\n🎉 批次 {batch_count} 完成，總等待 {total_wait}s (含1分鐘批次休息)")
                            else:
                                print(f"✅ #{len(self.results)} 完成，等待 {total_wait}s...")
                            
                            pbar.update(1)
                            pbar.set_postfix({
                                '並發': active_tasks, 
                                '本輪': results_processed,
                                '總完成': len(self.results),
                                '批次': batch_count % 100 if batch_count % 100 != 0 else 100
                            })
                            time.sleep(total_wait)
                    
                    for future in done_futures:
                        futures.remove(future)
                    
                    time.sleep(0.1)
                
                pbar.close()
                
        except KeyboardInterrupt:
            print("\n⏸️  Ctrl+C 中斷，保存進度...")

        print(f"\n✅ 本輪新增: {results_processed} 個結果")
        print(f"📊 總完成: {len(self.results)} 個結果")
        
        program_end = time.perf_counter()
        self.program_total_time = program_end - program_start
        print(f"⏱️  本輪時間: {self.program_total_time:.1f}s")

    def to_json_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def generate_report(self):
        """生成完整報告"""
        df = pd.DataFrame(self.results)
        if df.empty:
            print("❌ 無測試結果")
            return

        df['upload_time'] = pd.to_numeric(df.get('upload_time', 0.0), errors='coerce').fillna(0.0)
        df['analyze_time'] = pd.to_numeric(df.get('analyze_time', 0.0), errors='coerce').fillna(0.0)
        df['total_time'] = df['upload_time'] + df['analyze_time']

        real_df = df[df['label'] == 'bonafide']
        fake_df = df[df['label'] == 'spoof']

        real_scores = real_df['prob_real'].values if len(real_df) > 0 else np.array([])
        fake_scores = fake_df['prob_real'].values if len(fake_df) > 0 else np.array([])

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

        print("\n📊 完整統計（純處理時間，不含等待）:")
        print(f"總檔案數: {time_stats['files_count']}")
        print(f"⏱️ 平均單檔: {time_stats['avg_time']:.3f}s")
        print(f"📤 平均上傳: {time_stats['upload_avg']:.3f}s")
        print(f"⚙️  平均分析: {time_stats['analyze_avg']:.3f}s")

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

        df.to_csv(RESULTS_CSV, index=False)
        with open(SUMMARY_JSON, 'w') as f:
            json.dump(summary, f, indent=2, default=self.to_json_serializable)

        print(f"✅ CSV保存: {RESULTS_CSV}")
        print(f"✅ JSON保存: {SUMMARY_JSON}")
        print(f"🎯 EER: {eer:.10f} | AUC: {auc_score:.10f}")
        print("\n" + "="*60)
        print("PLATFORM EVALUATION COMPLETE")
        print("="*60)

if __name__ == "__main__":
    evaluator = PlatformEvaluator()

    print("🚀 ASVspoof 平台評估系統（斷點續傳 + 智能等待）")
    print(f"📁 FLAC: {FLAC_DIR}")
    print(f"📁 資料集: {DATASET_CSV}")
    print(f"📁 斷點: {CHECKPOINT_CSV}")
    print("-" * 50)

    # 10並發，智能等待（每檔2s + 每100檔1分鐘）
    evaluator.run_evaluation(max_files=100, max_concurrent=10, cooldown_sec=2)
    evaluator.generate_report()

