"""Pure pursuit 조향. 속도는 부르는 쪽에서 정한다."""
import numpy as np


def pure_pursuit(wp: np.ndarray, lookahead: float, wheelbase: float, steer_max: float) -> float:
    wp = np.asarray(wp, float)
    d = np.hypot(wp[:, 0], wp[:, 1])
    ahead = np.where(d >= lookahead)[0]
    i = int(ahead[0]) if len(ahead) else len(wp) - 1     # 전부 lookahead 안쪽이면 제일 먼 점
    x, y = wp[i]
    L2 = max(x * x + y * y, 1e-6)
    curvature = 2.0 * y / L2
    return float(np.clip(np.arctan(wheelbase * curvature), -steer_max, steer_max))
