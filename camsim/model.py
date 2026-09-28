"""작은 CNN 과 predict 래퍼.

시뮬 폐루프와 실차 ROS 노드가 똑같이 Predictor.predict(bev) 를 부른다. 그게 이 설계의 목표다.
"""
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
from .config import Config
from .track import Track
from . import gt
from .dataset import to_tensor
from .render import ipm_bev


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class WaypointNet(nn.Module):
    """stride 2 블록 5개로 줄이고 head 에서 waypoint 좌표를 뽑는다.

    AdaptiveAvgPool 을 써서 BEV 해상도를 바꿔도 head 크기는 그대로다. 편하지만 부작용이 있는데,
    학습 때와 다른 해상도를 넣어도 에러 없이 돌아가고 출력만 엉망이 된다. config 를 맞춰야 하는 이유.
    """

    def __init__(self, n_out: int = 2):
        super().__init__()
        self.features = nn.Sequential(_block(3, 16), _block(16, 32), _block(32, 64),
                                      _block(64, 128), _block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(),
                                  nn.Linear(128 * 8, 128), nn.ReLU(inplace=True),
                                  nn.Linear(128, n_out))

    def forward(self, x):
        return self.head(self.features(x))


class Predictor:
    def __init__(self, net: nn.Module, cfg: Config, device: str = "cpu"):
        self.net, self.cfg, self.device = net.to(device).eval(), cfg, device
        self.k = len(cfg.waypoints.ahead_m)

    @torch.no_grad()
    def predict(self, bev_bgr: np.ndarray) -> np.ndarray:
        """BEV -> waypoints (K,2) m. 시뮬과 실차가 공유하는 유일한 인터페이스."""
        x = to_tensor(bev_bgr)[None].to(self.device)
        y = self.net(x)[0].cpu().numpy() * self.cfg.waypoints.norm_m
        return y.reshape(self.k, 2)

    def predict_camera(self, cam_bgr: np.ndarray, H_i2g: np.ndarray) -> np.ndarray:
        """실차용. 카메라 영상을 IPM 으로 펴서 predict 에 넘긴다."""
        return self.predict(ipm_bev(cam_bgr, H_i2g, self.cfg))


class OraclePredictor:
    """모델 대신 정답을 그대로 돌려준다. 학습된 모델이 없어도 폐루프를 돌려볼 수 있고,
    noise_sigma 를 올려서 "인지 오차가 이만큼이면 주행이 어디서 깨지나"를 볼 수도 있다."""

    def __init__(self, track: Track, cfg: Config, noise_sigma: float = 0.0, rng=None):
        self.track, self.cfg, self.sigma = track, cfg, noise_sigma
        self.rng = rng or np.random.default_rng(0)
        self.pose = None

    def set_pose(self, pose):
        self.pose = np.asarray(pose, float)

    def predict(self, img_bgr) -> np.ndarray:
        wp = gt.waypoints_ahead(self.pose, self.track, self.cfg)
        if self.sigma > 0:
            wp = wp + self.rng.normal(0, self.sigma, wp.shape)
        return wp


def _input_spec(cfg: Config) -> dict:
    """Training settings needed to interpret a BEV and its waypoint outputs."""
    return {"bev": asdict(cfg.bev), "waypoints": asdict(cfg.waypoints),
            "lane_colors": {"floor": cfg.lane.color_floor, "tape": cfg.lane.color_tape}}


def save(net: nn.Module, path, cfg: Config = None) -> None:
    checkpoint = {"state_dict": net.state_dict(), "n_out": net.head[-1].out_features}
    if cfg is not None:
        checkpoint["input_spec"] = _input_spec(cfg)
    torch.save(checkpoint, path)


def load(path, cfg: Config) -> WaypointNet:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    expected = 2 * len(cfg.waypoints.ahead_m)
    if ck["n_out"] != expected:
        raise ValueError(
            f"checkpoint n_out={ck['n_out']} does not match cfg 2*len(waypoints.ahead_m)={expected}"
        )
    if "input_spec" in ck and ck["input_spec"] != _input_spec(cfg):
        raise ValueError("checkpoint input_spec does not match BEV, waypoint or lane color config")
    net = WaypointNet(n_out=ck["n_out"])
    net.load_state_dict(ck["state_dict"])
    return net
