#!/usr/bin/env python3
"""
ASVspoof2019 LA.cm.eval.trl.txt → 隨機選100檔案 + 標籤對照 (50 real + 50 fake)
"""

import pandas as pd
import random
import os
from pathlib import Path

# 配置路徑
PROTOCOL_FILE = "/home/carmen/asvspoof/datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
FLAC_DIR = Path("/home/carmen/asvspoof/datasets/LA/ASVspoof2019_LA_eval/flac")
OUTPUT_CSV = "eval_200_dataset.csv"  # 🔄 改名避免覆蓋

def parse_protocol_file(protocol_path):
    """解析 ASVSpoof2019.LA.cm.eval.trl.txt"""
    data = []

    print("📖 解析標籤檔...")
    with open(protocol_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) >= 4:
                speaker_id = parts[0]
                audio_id = parts[1]
                attack_id = parts[3] if len(parts) > 3 else "-"

                label = 'real' if attack_id == '-' else 'fake'
                filename = f"{audio_id}.flac"
                data.append({
                    'filename': filename,
                    'speaker_id': speaker_id,
                    'audio_id': audio_id,
                    'attack_id': attack_id,
                    'label': label
                })

    df = pd.DataFrame(data)
    print(f"✅ 解析完成: {len(df)} 個檔案")
    print(f"   真實聲音 (real): {len(df[df['label']=='real'])}")
    print(f"   虛假聲音 (fake): {len(df[df['label']=='fake'])}")
    return df

def select_random_200(df):  # 🔄 改名 + 改數字
    """隨機選100個檔案 (50 real + 50 fake)"""
    print("\n🎲 隨機選取100個檔案 (50+50)...")

    real_count = len(df[df['label']=='real'])
    fake_count = len(df[df['label']=='fake'])

    # 🔄 修改這兩行：50 real + 50 fake
    real_target = min(100, real_count)   # 改成 50
    fake_target = min(100, fake_count)   # 改成 50

    real_sample = df[df['label']=='real'].sample(n=real_target, random_state=42)
    fake_sample = df[df['label']=='fake'].sample(n=fake_target, random_state=42)

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
    selected_df = select_random_200(df)      # 🔄 使用新函數
    verified_df = verify_files(selected_df, FLAC_DIR)
    
    verified_df[['filename', 'label']].to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 測試清單已儲存: {OUTPUT_CSV}")
    print("\n📊 最終測試集統計:")
    print(verified_df['label'].value_counts())
    print("\n✅ 準備完成！使用 eval_100_dataset.csv")

if __name__ == "__main__":
    main()

