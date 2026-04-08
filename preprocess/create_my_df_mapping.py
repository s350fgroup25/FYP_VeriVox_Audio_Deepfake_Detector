#!/usr/bin/env python3
"""
掃描實際存在的1000個FLAC檔案 + 從完整metadata.txt配對標籤
生成 df_my_actual_1000_mapping.csv (只包含你真正擁有的檔案)
"""

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

# 配置路徑
FLAC_DIR = Path("/home/carmen/asvspoof/asvspoof2021")
METADATA_FILE = "/home/carmen/asvspoof/asvspoof2021/keys/DF/CM/trial_metadata.txt"
OUTPUT_CSV = "/home/carmen/asvspoof/preprocess/df_my_actual_1000_mapping.csv"

def scan_actual_flac_files(flac_dir):
    """掃描實際存在的FLAC檔案"""
    print("🔍 掃描實際存在的FLAC檔案...")
    
    actual_files = []
    flac_count = 0
    
    for root, dirs, files in os.walk(flac_dir):
        for file in files:
            if file.endswith('.flac'):
                flac_count += 1
                # 提取檔名 (DF_E_2000011.flac → DF_E_2000011)
                audio_id = file[:-5]  # 去掉 .flac
                actual_files.append(audio_id)
                if flac_count % 100 == 0:
                    print(f"   已掃描 {flac_count} 個FLAC檔案...")
    
    print(f"✅ 找到 {len(actual_files)} 個實際存在的FLAC檔案")
    return set(actual_files)

def parse_metadata(metadata_path):
    """解析完整metadata.txt，建立 audio_id → metadata 映射"""
    print("📖 解析完整metadata.txt...")
    
    metadata_map = {}
    with open(metadata_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) >= 4:
                speaker_id = parts[0]
                audio_id = parts[1]        # DF_E_2000011
                codec = parts[2]
                attack_type = parts[3]
                
                # label判斷：attack_type= '-' → real，否則 fake
                label = 'real' if attack_type == '-' else 'fake'
                
                metadata_map[audio_id] = {
                    'speaker_id': speaker_id,
                    'codec': codec,
                    'attack_type': attack_type,
                    'label': label
                }
    
    print(f"✅ metadata解析完成: {len(metadata_map)} 個記錄")
    return metadata_map

def create_mapping(actual_files, metadata_map):
    """配對實際檔案與metadata"""
    print("\n🔗 配對實際檔案與標籤...")
    
    matched_data = []
    unmatched_files = []
    
    for audio_id in actual_files:
        filename = f"{audio_id}.flac"
        
        if audio_id in metadata_map:
            metadata = metadata_map[audio_id]
            matched_data.append({
                'filename': filename,
                'audio_id': audio_id,
                'speaker_id': metadata['speaker_id'],
                'codec': metadata['codec'],
                'attack_type': metadata['attack_type'],
                'label': metadata['label']
            })
        else:
            unmatched_files.append(filename)
    
    print(f"✅ 成功配對: {len(matched_data)} 個檔案")
    print(f"❌ 未找到metadata: {len(unmatched_files)} 個檔案")
    
    if unmatched_files:
        print("未配對檔案前10個:", unmatched_files[:10])
    
    return matched_data, unmatched_files

def save_mapping(matched_data, output_csv):
    """儲存完整對焦表"""
    df = pd.DataFrame(matched_data)
    
    # 統計
    real_count = len(df[df['label']=='real'])
    fake_count = len(df[df['label']=='fake'])
    
    # 儲存完整資訊
    df.to_csv(output_csv, index=False)
    
    print(f"\n📊 **最終對焦表統計**")
    print(f"總計檔案: {len(df)}")
    print(f"真實 (real): {real_count}")
    print(f"假的 (fake): {fake_count}")
    print(f"比例: real:{real_count}/{len(df)} ({real_count/len(df)*100:.1f}%)")
    
    print(f"\n💾 完整對焦表已儲存: {output_csv}")
    
    # 也儲存簡化版 (只給evaluation用)
    simple_csv = output_csv.replace('.csv', '_simple.csv')
    df[['filename', 'label']].to_csv(simple_csv, index=False)
    print(f"💾 簡化版 (evaluation用): {simple_csv}")
    
    return df

def main():
    print("🚀 開始建立你的DF dataset真實對焦表")
    print(f"📁 FLAC目錄: {FLAC_DIR}")
    print(f"📄 Metadata: {METADATA_FILE}")
    
    # 1. 掃描實際檔案
    actual_files = scan_actual_flac_files(FLAC_DIR)
    
    # 2. 解析metadata
    metadata_map = parse_metadata(METADATA_FILE)
    
    # 3. 配對
    matched_data, unmatched = create_mapping(actual_files, metadata_map)
    
    # 4. 儲存
    df = save_mapping(matched_data, OUTPUT_CSV)
    
    print("\n🎉 **完成！**")
    print(f"✅ 你擁有 {len(df)} 個可用的配對檔案")
    print(f"📋 使用 df_my_actual_1000_mapping_simple.csv 進行evaluation")

if __name__ == "__main__":
    main()

