import os
import json
import pandas as pd
from src.data.loader import load_wptt_page_with_structure

def build_char_map(metadata_file, output_file):
    df = pd.read_csv(metadata_file, encoding='utf-8-sig')
    char_set = set()
    for _, row in df.iterrows():
        try:
            data = load_wptt_page_with_structure(row['file'])
        except:
            continue
        for ci in data['char_instances']:
            char_set.add(ci['char'])
    char_list = sorted(char_set)
    char2id = {ch: i for i, ch in enumerate(char_list)}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(char2id, f, ensure_ascii=False)
    print(f"字符数: {len(char2id)}，已保存到 {output_file}")

if __name__ == '__main__':
    metadata = 'data/features/metadata.csv'
    output = 'data/features/char_map.json'
    build_char_map(metadata, output)