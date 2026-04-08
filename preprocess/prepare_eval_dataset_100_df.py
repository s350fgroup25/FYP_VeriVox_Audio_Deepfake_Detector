#!/usr/bin/env python3
"""
ASVspoof2021 DF/cm/trial_metadata.txt → 隨機選1000檔案 + 標籤對照 (平衡real/spoof)
"""

import pandas as pd
import random
import os
from pathlib import Path

# 🔄 2021 DF dataset配置
PROTOCOL_FILE = "/home/carmen/asvspoof/asvspoof2021/keys/DF/CM/trial_metadata.txt"
FLAC_DIR = Path("/home/carmen/asvspoof/asvspoof2021")
OUTPUT_CSV = "df_eval_1000_dataset.csv"  # 目標輸出檔案

def parse_protocol_file(protocol_path):
    """解析ASVspoof2021 DF trial_metadata.txt格式"""
    data = []
    
    print("📖 解析2021 DF標籤檔...")
    with open(protocol_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) >= 4:
                speaker_id = parts[0]      # LA_0023, TEF2等
                audio_id = parts[1]        # DF_E_2000011等
                codec = parts[2]           # nocodec, low_m4a等
                attack_type = parts[3]     # asvspoof, vcc2020等
                
                # 判斷label：根據格式，spoof有attack_type，real是"-"
                label = 'real' if attack_type == '-' else 'fake'
                
                # 提取純檔名（去掉DF_前綴）
                filename = audio_id + ".flac"  # DF_E_2000011.flac
                
                data.append({
                    'filename': filename,
                    'speaker_id': speaker_id,
                    'audio_id': audio_id,
                    'codec': codec,
                    'attack_type': attack_type,
                    'label': label
                })
    
    df = pd.DataFrame(data)
    print(f"✅ 解析完成: {len(df)} 個檔案")
    print(f"   真實聲音 (real): {len(df[df['label']=='real'])}")
    print(f"   虛假聲音 (fake): {len(df[df['label']=='fake'])}")
    return df

def select_random_1000(df):
    """隨機選1000個檔案 (盡量平衡real/fake)"""
    print("\n🎲 隨機選取1000個檔案...")
    
    real_df = df[df['label']=='real']
    fake_df = df[df['label']=='fake']
    
    # 目標各500個（如果不足就全取）
    real_target = min(500, len(real_df))
    fake_target = min(500, len(fake_df))
    
    real_sample = real_df.sample(n=real_target, random_state=42)
    fake_sample = fake_df.sample(n=fake_target, random_state=42)
    
    selected_df = pd.concat([real_sample, fake_sample]).reset_index(drop=True)
    print(f"✅ 選取完成:")
    print(f"   真實聲音: {len(real_sample)}")
    print(f"   虛假聲音: {len(fake_sample)}")
    print(f"   總計: {len(selected_df)}")
    return selected_df

def verify_files(df, flac_dir):
    """驗證檔案是否存在"""
    print("\n🔍 驗證檔案存在性...")
    existing_files = []
    
    for _, row in df.iterrows():
        filepath = flac_dir / row['filename']
        if filepath.exists():
            existing_files.append(row)
        else:
            print(f"❌ 找不到: {row['filename']}")
    
    verified_df = pd.DataFrame(existing_files)
    print(f"✅ 可使用檔案: {len(verified_df)}")
    return verified_df

def main():
    df = parse_protocol_file(PROTOCOL_FILE)
    selected_df = select_random_1000(df)
    verified_df = verify_files(selected_df, FLAC_DIR)
    
    # 只保留必要欄位給evaluation使用
    verified_df[['filename', 'label']].to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 測試清單已儲存: {OUTPUT_CSV}")
    print("\n📊 最終測試集統計:")
    print(verified_df['label'].value_counts())
    print(f"\n✅ 準備完成！使用 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

