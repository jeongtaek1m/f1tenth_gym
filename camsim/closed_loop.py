"""gym 안에서 렌더 -> 추론 -> pure pursuit -> step 을 한 루프로 돌린다. ROS 는 안 쓴다.

종료 사유(Result.reason)
  lap          한 바퀴 완주
  tape_crossed 차체 모서리가 테이프 안쪽 선을 넘음. 실차와 같은 실격 규칙이다.
  collision    gym 벽 충돌. 이 맵은 벽이 테이프보다 훨씬 멀어서 거의 안 난다.
  max_steps    시간 초과

실차 트랙에는 벽이 없고 테이프가 경계라서, 실격 판정도 벽이 아니라 테이프로 한다.
"""
from collections import deque
from dataclasses import dataclass
import os
import warnings
import cv2
import numpy as np
from .config import Config
from .track import Track
from . import gt, render, viz
from .pure_pursuit import pure_pursuit
from .model import OraclePredictor


def make_env(cfg: Config):
    import gym
    from f110_gym.envs.base_classes import Integrator
    base = os.path.splitext(cfg.closed_loop.map_yaml)[0]
    with warnings.catch_warnings():
        # gym 이 RK4 를 고를 때마다 경고를 뱉는데, 우리가 일부러 고른 거라 조용히 시킨다.
        warnings.filterwarnings("ignore", message="Chosen integrator is RK4.*")
        return gym.make("f110_gym:f110-v0", map=base, map_ext=".png", num_agents=1,
                        timestep=0.01, integrator=Integrator.RK4)


def pose_of(obs) -> np.ndarray:
    return np.array([obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]])


@dataclass
class Result:
    finished: bool
    reason: str
    steps: int
    mean_lateral_m: float
    max_lateral_m: float
    progress_m: float
    control_hz_eff: float
    time_s: float = 0.0                # 주행 시간(초). 랩타임 비교용
    lateral_trace: np.ndarray = None   # 제어 틱마다의 횡오차(m). 그래프용
    pose_trace: np.ndarray = None      # (steps,3) 제어 틱마다의 world pose. 맵에 경로 그릴 때


def _unwrap_progress(track: Track, s_prev: float, s_now: float) -> float:
    # 결승선을 넘어가면 s 가 length -> 0 으로 점프한다. 그 점프를 정상적인 전진으로 되돌린다.
    ds = s_now - s_prev
    if ds < -track.length / 2: ds += track.length
    if ds > track.length / 2: ds -= track.length
    return ds


def run(env, predictor, track: Track, cfg: Config, H_g2i: np.ndarray, start_index: int = 0,
        latency_steps=None, video_path=None) -> Result:
    cl = cfg.closed_loop
    if latency_steps is None:
        latency_steps = cl.latency_steps

    # 제어 한 틱마다 물리를 몇 번 돌릴지. control_hz 가 물리 주파수의 약수가 아니면 반올림되면서
    # 실제 제어 주기가 요청값과 달라지므로, 그 경우 경고하고 실제값(hz_eff)을 결과에 남긴다.
    physics_per_tick = max(1, int(round(1.0 / (env.timestep * cl.control_hz))))
    hz_eff = 1.0 / (env.timestep * physics_per_tick)
    if abs(hz_eff - cl.control_hz) / cl.control_hz > 0.02:
        warnings.warn(
            f"camsim: control_hz={cl.control_hz} is not an integer divisor of the gym physics "
            f"rate (1/{env.timestep}); effective control rate is {hz_eff:.3f} Hz instead",
            UserWarning,
        )

    p0 = track.center[start_index]
    obs, _, done, _ = env.reset(np.array([[p0[0], p0[1], track.heading[start_index]]]))
    # 지연 버퍼. 처음 latency_steps 틱 동안은 "직진" 예측을 내보낸다.
    buf = deque([np.array([cfg.waypoints.ahead_m, 0.0])] * latency_steps)

    mask = render.bev_visibility_mask(H_g2i, cfg)
    writer = None            # 영상은 [카메라 뷰 | 모델 입력 BEV]. 첫 프레임 크기로 열린다.

    lats, traveled, reason = [], 0.0, "max_steps"
    poses = []
    s_prev = track.s[gt.nearest_index(track, p0)]
    steps = 0
    try:
        for steps in range(1, cl.max_steps + 1):
            pose = pose_of(obs)
            bev = render.render_bev(pose, track.quads, cfg, mask)
            if hasattr(predictor, "set_pose"):
                predictor.set_pose(pose)
            buf.append(predictor.predict(bev))
            wp = buf.popleft()                     # latency_steps 틱 전의 예측으로 조향한다
            steer = pure_pursuit(wp, cl.wheelbase_m, cl.steer_max_rad)
            if video_path is not None:
                cam = render.draw_points(render.render(pose, track.quads, obs["scans"][0], H_g2i, cfg), wp, H_g2i)
                frame = viz.side_by_side(cam, render.draw_points_bev(bev.copy(), wp, cfg))
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), hz_eff,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            for _ in range(physics_per_tick):
                obs, _, done, _ = env.step(np.array([[steer, cl.speed_mps]]))
                if done:
                    break
            pose = pose_of(obs)
            lat = gt.lateral_error(track, pose[:2])
            lats.append(lat)
            poses.append(pose)
            s_now = track.s[gt.nearest_index(track, pose[:2])]
            traveled += _unwrap_progress(track, s_prev, s_now)
            s_prev = s_now
            if obs["collisions"][0]:
                reason = "collision"; break
            if gt.crosses_tape(pose, track, cfg):
                reason = "tape_crossed"; break
            if traveled >= track.length:
                reason = "lap"; break
    finally:
        if writer is not None:                     # 중간에 터져도 mp4 는 닫아야 재생된다
            writer.release()
    lats = np.array(lats) if lats else np.zeros(1)
    return Result(reason == "lap", reason, steps, float(lats.mean()), float(lats.max()), float(traveled),
                  hz_eff, steps / hz_eff, lats, np.array(poses).reshape(-1, 3))


def sweep(env, track: Track, cfg: Config, H_g2i, latency_list, sigma_list, predictor_factory=None):
    """지연 x 인지오차 격자를 훑는다. predictor_factory 를 안 주면 오라클에 노이즈를 섞어 쓴다."""
    rows = []
    for lat in latency_list:
        for sig in sigma_list:
            pred = (predictor_factory(sig) if predictor_factory
                    else OraclePredictor(track, cfg, noise_sigma=sig))
            r = run(env, pred, track, cfg, H_g2i, latency_steps=lat)
            rows.append({"latency_steps": lat, "sigma": sig, "finished": r.finished, "reason": r.reason,
                         "mean_lateral_m": r.mean_lateral_m, "max_lateral_m": r.max_lateral_m,
                         "steps": r.steps, "progress_m": r.progress_m})
            print(f"latency={lat:2d} sigma={sig:.2f} -> {r.reason:9s} lat_mean={r.mean_lateral_m:.3f} "
                  f"lat_max={r.max_lateral_m:.3f} progress={r.progress_m:.1f}m", flush=True)
    return rows
