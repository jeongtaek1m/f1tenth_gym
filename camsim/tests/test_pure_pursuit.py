import numpy as np, pytest
from camsim import config
from camsim.pure_pursuit import pure_pursuit

_cfg = config.load()
WB, SMAX = _cfg.closed_loop.wheelbase_m, _cfg.closed_loop.steer_max_rad

def test_straight_gives_zero():
    assert pure_pursuit(np.array([[1.0, 0.0]]), WB, SMAX) == pytest.approx(0.0)

def test_left_curve_positive_and_clipped():
    assert 0 < pure_pursuit(np.array([[1.0, 0.3]]), WB, SMAX) <= SMAX
    assert pure_pursuit(np.array([[0.5, 0.5]]), WB, SMAX) == pytest.approx(SMAX)

def test_right_curve_negative():
    assert pure_pursuit(np.array([[1.0, -0.3]]), WB, SMAX) < 0

def test_multiple_waypoints_steer_to_last():
    wp = np.array([[0.5, -0.5], [1.0, 0.3]])      # 앞 점은 오른쪽, 마지막 점은 왼쪽
    assert pure_pursuit(wp, WB, SMAX) > 0

def test_accepts_flat_xy():
    assert pure_pursuit([1.0, 0.0], WB, SMAX) == pytest.approx(0.0)
