"""Pure pursuit 조향. 속도는 부르는 쪽에서 정한다."""
import numpy as np


def pure_pursuit(wp, wheelbase: float, steer_max: float) -> float:
    """예측 waypoint 로 조향각을 낸다. 점이 여러 개면 마지막(가장 먼) 점을 목표로 삼는다.

    목표점까지의 거리가 곧 lookahead 라서 따로 설정값이 없다. waypoints.ahead_m 이 그 역할이다.
    """
    x, y = np.asarray(wp, float).reshape(-1, 2)[-1]
    L2 = max(x * x + y * y, 1e-6)
    curvature = 2.0 * y / L2
    return float(np.clip(np.arctan(wheelbase * curvature), -steer_max, steer_max))
