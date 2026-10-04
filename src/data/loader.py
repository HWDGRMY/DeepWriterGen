import os
import struct
import numpy as np


def load_wptt_page(file_path):
    """
    基于 CASIA 官方 WPTT 结构解析单页文本轨迹。
    严格遵循 CSDN 文章文件头结构解析。
    """
    all_points = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'rb') as f:
        # ================= 1. 解析 WPTT 文件头 =================
        # 头部总长度 (4B, 小端序 uint32)
        header_len = struct.unpack('<I', f.read(4))[0]

        # Format code (8B, 固定 'WPTT')
        fmt = f.read(8)
        if not fmt.startswith(b'WPTT'):
            raise ValueError(f"文件格式错误，期望 WPTT，实际得到 {fmt}")

        # Illustration (可变长字符串，以 \0 结尾)
        illustration = b''
        while True:
            c = f.read(1)
            if not c or c == b'\0':
                break
            illustration += c

        # Code type (20B, 固定 'GB')
        code_type = f.read(20)
        # Code length (2B, 短整型, 固定 2)
        code_len = struct.unpack('<H', f.read(2))[0]
        # Data type (20B, 固定 'short')
        data_type = f.read(20)

        # Sample length (4B, 文件总字节数)
        sample_len = struct.unpack('<I', f.read(4))[0]
        # Page index (4B, 页码)
        page_idx = struct.unpack('<I', f.read(4))[0]
        # Stroke number (4B, 当前页所有笔画总数)
        stroke_num = struct.unpack('<I', f.read(4))[0]

        # ================= 2. 读取全部笔画 =================
        all_strokes = []
        for _ in range(stroke_num):
            # 单笔画包含的点数 (2B)
            pt_count = struct.unpack('<H', f.read(2))[0]
            stroke_pts = []
            for _ in range(pt_count):
                # 坐标 X (2B, 有符号短整型) -> 实际坐标需除以 10
                x = struct.unpack('<h', f.read(2))[0] / 10.0
                # 坐标 Y (2B, 有符号短整型)
                y = struct.unpack('<h', f.read(2))[0] / 10.0
                stroke_pts.append((x, y))
            all_strokes.append(stroke_pts)

        # ================= 3. 读取文本行与字符编码 =================
        # 文本行总数 (2B)
        line_num = struct.unpack('<H', f.read(2))[0]

        for _ in range(line_num):
            # 该行的笔画数量 (2B)
            line_stroke_num = struct.unpack('<H', f.read(2))[0]
            # 该行包含的笔画索引数组 (2B * line_stroke_num)
            line_stroke_idx = []
            for _ in range(line_stroke_num):
                idx = struct.unpack('<H', f.read(2))[0]
                line_stroke_idx.append(idx)

            # 该行的字符数量 (2B)
            line_char_num = struct.unpack('<H', f.read(2))[0]
            # 字符 GB 编码 (2B * line_char_num)
            line_chars = []
            for _ in range(line_char_num):
                # 固定 2 字节 GB2312 编码
                tag_code = f.read(code_len)
                # 用 GBK 解码（GB2312 的超集），处理乱码
                char = tag_code.decode('gbk', errors='ignore')
                line_chars.append(char)

    # ================= 4. 拼装为论文所需的轨迹序列 =================
    for stroke in all_strokes:
        if not stroke:
            continue
        for pt in stroke:
            # 过滤掉坐标原点(0,0)附近 2.0 以内的孤立噪点
            if abs(pt[0]) <= 2.0 and abs(pt[1]) <= 2.0:
                continue
            all_points.append(pt)
        all_points.append((-1.0, 0.0))

    return np.array(all_points, dtype=np.float32)

def load_wptt_page_with_structure(file_path):
    """
    解析 WPTT 文件，返回整页轨迹、笔画、行结构（字符+笔画索引）。
    """
    import struct
    import numpy as np

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'rb') as f:
        header_len = struct.unpack('<I', f.read(4))[0]
        fmt = f.read(8)
        if not fmt.startswith(b'WPTT'):
            raise ValueError(f"文件格式错误，期望 WPTT，实际得到 {fmt}")
        illustration = b''
        while True:
            c = f.read(1)
            if not c or c == b'\0':
                break
            illustration += c
        code_type = f.read(20)
        code_len = struct.unpack('<H', f.read(2))[0]
        data_type = f.read(20)
        sample_len = struct.unpack('<I', f.read(4))[0]
        page_idx = struct.unpack('<I', f.read(4))[0]
        stroke_num = struct.unpack('<I', f.read(4))[0]

        strokes = []
        for _ in range(stroke_num):
            pt_count = struct.unpack('<H', f.read(2))[0]
            stroke_pts = []
            for _ in range(pt_count):
                x = struct.unpack('<h', f.read(2))[0] / 10.0
                y = struct.unpack('<h', f.read(2))[0] / 10.0
                stroke_pts.append((x, y))
            strokes.append(stroke_pts)

        line_num = struct.unpack('<H', f.read(2))[0]
        lines = []
        for _ in range(line_num):
            line_stroke_num = struct.unpack('<H', f.read(2))[0]
            line_stroke_idx = []
            for _ in range(line_stroke_num):
                idx = struct.unpack('<H', f.read(2))[0]
                line_stroke_idx.append(idx)
            line_char_num = struct.unpack('<H', f.read(2))[0]
            line_chars = []
            for _ in range(line_char_num):
                tag_code = f.read(code_len)
                char = tag_code.decode('gbk', errors='ignore')
                line_chars.append(char)
            lines.append({
                'stroke_indices': line_stroke_idx,
                'chars': line_chars,
            })

    # 组装整页轨迹（所有笔画点，用 -1 分隔）
    all_points = []
    for stroke in strokes:
        for pt in stroke:
            if abs(pt[0]) <= 2.0 and abs(pt[1]) <= 2.0:
                continue
            all_points.append(pt)
        all_points.append((-1.0, 0.0))

    # 为每个字符提取轨迹和 bbox
    char_instances = []  # 每个元素：{'char': 字符, 'points': np.array, 'bbox': (h,w,v_center,h_offset)}
    for line in lines:
        stroke_indices = line['stroke_indices']
        chars = line['chars']
        if len(stroke_indices) != len(chars):
            continue
        for idx, char in zip(stroke_indices, chars):
            if idx >= len(strokes):
                continue
            pts = np.array(strokes[idx], dtype=np.float32)
            if len(pts) < 3:
                continue
            # 计算 bbox
            min_x, max_x = pts[:,0].min(), pts[:,0].max()
            min_y, max_y = pts[:,1].min(), pts[:,1].max()
            w = max_x - min_x
            h = max_y - min_y
            v_center = (min_y + max_y) / 2
            # 水平偏移相对于前一个字符（由调用者计算）
            char_instances.append({
                'char': char,
                'points': pts,
                'bbox': (h, w, v_center, 0.0),  # h_offset 稍后计算
                'min_x': min_x, 'max_x': max_x,
                'min_y': min_y, 'max_y': max_y,
            })

    # 计算每个字符相对于前一个的水平偏移
    for i in range(1, len(char_instances)):
        prev_max_x = char_instances[i-1]['max_x']
        cur_min_x = char_instances[i]['min_x']
        h_offset = cur_min_x - prev_max_x
        h, w, v_center, _ = char_instances[i]['bbox']
        char_instances[i]['bbox'] = (h, w, v_center, h_offset)

    # 重新计算整页点序列（保持原始顺序）
    all_points_np = np.array(all_points, dtype=np.float32)

    return {
        'all_points': all_points_np,
        'strokes': strokes,
        'lines': lines,
        'char_instances': char_instances,
        'page_idx': page_idx,
    }