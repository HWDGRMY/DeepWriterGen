"""
测试 Layout Generator：
  1. 用不同书写者的风格向量
  2. 对同样的文本预测换行
  3. 观察不同书写者是否有不同的换行习惯
"""
import os
import sys
import json
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from src.models.layout_generator import LayoutGenerator


def predict_lines(model, text, style_vec, char2id, device,
                  threshold=0.5):
    """预测文本的分行方案。"""
    ids = [char2id.get(c, 1) for c in text[:128]]
    ids += [0] * (128 - len(ids))
    ids_t = torch.tensor([ids]).to(device)
    sv = style_vec.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(ids_t, sv)[0]
        probs = torch.sigmoid(logits).cpu().numpy()

    lines, cur = [], []
    for i, ch in enumerate(text):
        if i >= len(probs):
            break
        cur.append(ch)
        if probs[i] > threshold:
            lines.append(''.join(cur))
            cur = []
    if cur:
        lines.append(''.join(cur))
    return lines, probs[:len(text)]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载字符表
    char2id = json.load(open('data/features/char_map.json', encoding='utf-8'))
    n = len(char2id)
    print(f"字符表: {n}")

    # 加载 Layout 模型
    model = LayoutGenerator(num_chars=n).to(device)
    ckpt = torch.load('outputs/checkpoints/layout_best.pth',
                      map_location=device)
    model.load_state_dict(ckpt['layout'])
    model.eval()
    print(f"Layout 模型已加载")

    # 加载风格向量
    sv_dict = torch.load('data/features/style_vectors.pt', map_location='cpu')
    print(f"风格向量: {len(sv_dict)} 位书写者\n")

    # 测试文本（挑一段常见内容的）
    test_texts = [
        '落霞与孤鹜齐飞秋水共长天一色',
        '床前明月光疑是地上霜举头望明月低头思故乡',
        '春眠不觉晓处处闻啼鸟夜来风雨声花落知多少',
    ]

    # 挑 3 位不同书写者
    test_writers = list(sv_dict.keys())[:3]
    print(f"测试书写者: {test_writers}\n")

    # 对每位书写者 × 每段文本预测换行
    for text in test_texts:
        print("=" * 70)
        print(f"文本: {text} (共 {len(text)} 字)")
        print("=" * 70)
        for wid in test_writers:
            sv = sv_dict[wid]
            lines, probs = predict_lines(model, text, sv, char2id, device)
            print(f"\n书写者 {wid}:")
            for i, lt in enumerate(lines):
                print(f"  行{i+1} ({len(lt)} 字): {lt}")

            # 打印换行概率曲线
            prob_str = ' '.join(f'{p:.2f}' for p in probs)
            print(f"  换行概率: {prob_str}")


if __name__ == '__main__':
    main()