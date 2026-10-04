import numpy as np

def resample_trajectory(points, seq_len=100):
    """
    将轨迹点序列重采样到固定长度 seq_len。
    points: (N, 2) 或 (N, 3) 其中第三列为 pen_state（可选）
    返回: (seq_len, 2) 或 (seq_len, 3)
    """
    if len(points) == 0:
        return np.zeros((seq_len, 2), dtype=np.float32)
    # 计算累积弧长
    diffs = np.diff(points[:, :2], axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    cumdist = np.concatenate([[0], np.cumsum(dists)])
    total = cumdist[-1]
    if total == 0:
        return np.tile(points[0:1, :2], (seq_len, 1)).astype(np.float32)
    # 目标弧长
    target = np.linspace(0, total, seq_len)
    # 插值
    new_x = np.interp(target, cumdist, points[:, 0])
    new_y = np.interp(target, cumdist, points[:, 1])
    resampled = np.stack([new_x, new_y], axis=1).astype(np.float32)
    return resampled


def compute_bbox_from_points(points):
    """从点序列计算 bbox: (h, w, v_center)"""
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    h = max_y - min_y
    w = max_x - min_x
    v_center = (min_y + max_y) / 2
    return h, w, v_center