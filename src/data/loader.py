import os
import struct
import numpy as np


def _decode_tag(raw):
    if len(raw) != 2:
        return ''
    b0, b1 = raw[0], raw[1]
    if b1 == 0 and 0 < b0 < 127:
        return chr(b0)
    if b0 == 0 and 0 < b1 < 127:
        return chr(b1)
    try:
        s = raw.decode('gbk')
        if s and s.isprintable():
            return s
    except Exception:
        pass
    try:
        s = raw[::-1].decode('gbk')
        if s and s.isprintable():
            return s
    except Exception:
        pass
    return ''


def load_wptt_page_with_structure(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, 'rb') as f:
        struct.unpack('<I', f.read(4))[0]
        fmt = f.read(8)
        if not fmt.startswith(b'WPTT'):
            raise ValueError(f"非 WPTT: {file_path}")
        while True:
            c = f.read(1)
            if not c or c == b'\0':
                break
        f.read(20)
        code_len = struct.unpack('<H', f.read(2))[0]
        f.read(20)
        struct.unpack('<I', f.read(4))[0]         # sample_len
        page_idx = struct.unpack('<I', f.read(4))[0]
        stroke_num = struct.unpack('<I', f.read(4))[0]

        strokes = []
        for _ in range(stroke_num):
            pt_count = struct.unpack('<H', f.read(2))[0]
            pts = []
            for _ in range(pt_count):
                x = struct.unpack('<h', f.read(2))[0] / 10.0
                y = struct.unpack('<h', f.read(2))[0] / 10.0
                pts.append((x, y))
            strokes.append(pts)

        line_num = struct.unpack('<H', f.read(2))[0]
        lines = []
        for _ in range(line_num):
            nstroke = struct.unpack('<H', f.read(2))[0]
            stroke_idx = [struct.unpack('<H', f.read(2))[0]
                          for _ in range(nstroke)]
            nchar = struct.unpack('<H', f.read(2))[0]
            chars = []
            for _ in range(nchar):
                ch = _decode_tag(f.read(code_len))
                if ch:
                    chars.append(ch)

            line_pts = []
            for si in stroke_idx:
                if si >= len(strokes):
                    continue
                for pt in strokes[si]:
                    if abs(pt[0]) <= 2.0 and abs(pt[1]) <= 2.0:
                        continue
                    line_pts.append(pt)
                line_pts.append((-1.0, 0.0))

            lines.append({
                'text': ''.join(chars),
                'points': np.array(line_pts, dtype=np.float32),
                'stroke_indices': stroke_idx,
            })

    return {'lines': lines, 'page_idx': page_idx}