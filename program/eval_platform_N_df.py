#!/usr/bin/env python3
"""
ASVspoof 2021 DF 平台真實用戶評估：精簡版 (只測試 upload + analyze)
適配2021 DF dataset路徑
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
DATASET_CSV = "/home/carmen/asvspoof/preprocess/df_my_actual_1000_mapping_simple.csv"
FLAC_DIR = Path("/home/carmen/asvspoof/asvspoof2021")
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
                # 2021標準：real→bonafide, fake→spoof
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

    # run_evaluation, to_json_serializable, generate_report 方法保持不變...
    # （為了節省空間，這裡省略，與原程式完全相同）

    def run_evaluation(self, max_files=1000, max_concurrent=1):
        """動態並發評估 - 保持原邏輯"""
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

    # generate_report 方法與原程式相同，略...

if __name__ == "__main__":
    evaluator = PlatformEvaluator()
    
    print("🚀 開始2021 DF評估流程...")
    print(f"📁 2021 DF FLAC路徑: {FLAC_DIR}")
    print(f"📁 CSV資料集: {DATASET_CSV}")
    print(f"📁 結果保存至: {RESULTS_CSV}")
    
    evaluator.run_evaluation(max_files=1000, max_concurrent=1)
    evaluator.generate_report()

