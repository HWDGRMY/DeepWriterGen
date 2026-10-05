import os
import sys
import json
import random
import glob
import pandas as pd
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from src.data.loader import load_wptt_page_with_structure


def build_metadata(raw_dir, out_dir, seed=42, n_train=700, n_val=150, n_test=169):
    random.seed(seed)
    files = glob.glob(os.path.join(raw_dir, '**', '*.wptt'), recursive=True)
    print(f"找到 {len(files)} 个 .wptt")

    writer_pages = defaultdict(list)
    for f in files:
        wid = os.path.basename(f).split('-')[0]
        writer_pages[wid].append({'file': f, 'page': os.path.basename(f).split('.')[0]})

    writers = sorted(writer_pages.keys())
    random.shuffle(writers)
    train_w = set(writers[:n_train])
    val_w = set(writers[n_train:n_train + n_val])
    test_w = set(writers[n_train + n_val:n_train + n_val + n_test])

    records = []
    for wid, pages in writer_pages.items():
        split = ('Train' if wid in train_w else
                 'Val' if wid in val_w else
                 'Test' if wid in test_w else None)
        if split is None:
            continue
        for p in pages:
            records.append({'file': p['file'], 'writer_id': wid,
                            'split': split, 'page': p['page']})

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(records)
    df['writer_id'] = df['writer_id'].astype(str)
    path = os.path.join(out_dir, 'metadata_gen.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    for split in ['Train', 'Val', 'Test']:
        sub = df[df['split'] == split]
        print(f"  {split}: {sub['writer_id'].nunique()} 位, {len(sub)} 页")
    return path


def build_char_map(metadata_file, output_file):
    df = pd.read_csv(metadata_file, encoding='utf-8-sig')
    counter = Counter()
    for _, row in df.iterrows():
        try:
            data = load_wptt_page_with_structure(row['file'])
        except Exception:
            continue
        for line in data['lines']:
            for ch in line['text']:
                counter[ch] += 1
    sorted_chars = [ch for ch, _ in counter.most_common()]
    char2id = {'<pad>': 0, '<unk>': 1}
    for ch in sorted_chars:
        if ch not in char2id:
            char2id[ch] = len(char2id)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(char2id, f, ensure_ascii=False, indent=1)
    print(f"字符表: {len(char2id)} 个")
    print(f"前 20: {sorted_chars[:20]}")
    return char2id


if __name__ == '__main__':
    raw = os.path.join(BASE_DIR, 'data', 'raw')
    out = os.path.join(BASE_DIR, 'data', 'features')
    meta = build_metadata(raw, out)
    build_char_map(meta, os.path.join(out, 'char_map.json'))