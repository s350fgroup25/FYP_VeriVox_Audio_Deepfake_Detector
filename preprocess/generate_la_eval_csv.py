#!/usr/bin/env python3
"""
從現有的1000個flac + trial_metadata.txt 產生對應CSV
只包含flac資料夾裡存在的檔案
"""

import csv
from pathlib import Path
import pandas as pd

FLAC_DIR = Path("/home/carmen/asvspoof/asvspoof2021/LA/flac")
META_FILE = "/home/carmen/asvspoof/asvspoof2021/LA/keys/LA/CM/trial_metadata.txt"
OUTPUT_CSV = "/home/carmen/asvspoof/preprocess/LA_1000_from_flac.csv"

def parse_meta_line(line):
    """解析LA trial_metadata格式"""
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    utt_id = parts[1]  # LA_E_xxxxxxx
    key = parts[5]     # bonafide/spoof (第6欄)
    filename = utt_id + ".flac"
    label = "real" if key == "bonafide" else "spoof"
    return filename, label

def main():
    print("🔍 掃描現有1000個flac...")
    
    # 1. 取得所有flac檔名
    flac_files = FLAC_DIR.glob("*.flac")
    flac_list = [f.name for f in flac_files]
    print(f"✅ 找到 {len(flac_list)} 個flac檔案")
    
    # 2. 讀取metadata，建立filename→label對照
    print("📖 解析trial_metadata.txt...")
    label_dict = {}
    
    with open(META_FILE, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 20000 == 0:
                print(f"   已處理 {line_num:,} 行")
            
            item = parse_meta_line(line)
            if item:
                filename, label = item
                label_dict[filename] = label
    
    print(f"✅ metadata解析完成: {len(label_dict)} 個檔案有標籤")
    
    # 3. 產生CSV (只包含存在的flac + 有標籤的)
    results = []
    missing_labels = 0
    
    for filename in flac_list:
        if filename in label_dict:
            results.append([filename, label_dict[filename]])
        else:
            print(f"⚠️  {filename} 無標籤")
            missing_labels += 1
    
    print(f"✅ 有效配對: {len(results)} (無標籤: {missing_labels})")
    
    # 4. 儲存CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(results)
    
    # 5. 統計
    df = pd.read_csv(OUTPUT_CSV)
    print(f"\n📊 最終統計:")
    print(df['label'].value_counts())
    print(f"💾 CSV儲存: {OUTPUT_CSV}")
    
    print("\n🎯 使用指令:")
    print(f"DATASET_CSV = \"{OUTPUT_CSV}\"")
    print('FLAC_DIR = Path("/home/carmen/asvspoof/asvspoof2021/LA/flac")')

if __name__ == "__main__":
    main()
