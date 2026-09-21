"""학습 루프와 오프라인 평가.

손실은 Huber(SmoothL1). waypoint 회귀라 가끔 튀는 라벨에 L2 보다 덜 흔들린다.
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import Config
from .track import Track
from .dataset import SynthDataset, DiskDataset, make_sample
from . import model as M


def evaluate(predictor, track: Track, cfg: Config, n: int = 200, seed: int = 123, degrade_fn=None) -> dict:
    """새로 뽑은 pose n 개에 대한 waypoint 별 오차(m).

    degrade_fn(bev, rng) 을 주면 열화된 입력에 대한 강건성 측정이 된다 (노트북 3장 sim-to-real 표).
    """
    from .camera import build
    from .render import bev_visibility_mask
    rng = np.random.default_rng(seed)
    mask = bev_visibility_mask(build(cfg)[0], cfg)
    errs = []
    for _ in range(n):
        bev, wp, pose = make_sample(track, cfg, rng, degrade_fn, mask=mask)
        if hasattr(predictor, "set_pose"):        # OraclePredictor 는 이미지가 아니라 pose 를 본다
            predictor.set_pose(pose)
        pred = predictor.predict(bev)
        errs.append(np.hypot(*(pred - wp).T))
    errs = np.array(errs)
    per = errs.mean(0)
    return {"per_waypoint_m": per, "mean_m": float(per.mean()), "max_m": float(errs.max())}


def _batches(dataset, batch_size, num_workers, seed):
    """무한 배치 스트림. DiskDataset 은 epoch 를 계속 돌려서 무한으로 만든다."""
    if isinstance(dataset, torch.utils.data.IterableDataset):
        yield from DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    else:
        g = torch.Generator().manual_seed(seed)
        while True:
            yield from DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  generator=g, drop_last=True)


@torch.no_grad()
def _val_loss(net, val_dataset, loss_fn, batch_size, val_batches, device):
    net.eval()
    dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    tot, k = 0.0, 0
    for x, y in dl:
        tot += loss_fn(net(x.to(device)), y.to(device)).item(); k += 1
        if k >= val_batches:
            break
    net.train()
    return tot / max(k, 1)


def train(track: Track, cfg: Config, steps: int, batch_size: int = 32, lr: float = 1e-3,
          device: str = "cpu", out_path=None, num_workers: int = 0, log_every: int = 50, seed: int = 0,
          dataset=None, val_dataset=None, val_batches: int = 8, callback=None):
    """dataset 을 안 주면 온더플라이(SynthDataset)로 학습한다.

    val_dataset 이 있으면 log_every 마다 val loss 도 재서 history 에 넣는다.
    callback(history) 는 log_every 마다 불린다. 노트북에서 loss 곡선을 실시간으로 그릴 때 쓴다.
    """
    torch.manual_seed(seed)
    net = M.WaypointNet(n_out=2 * len(cfg.waypoints.ahead_m)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = torch.nn.SmoothL1Loss()
    dl = _batches(dataset if dataset is not None else SynthDataset(track, cfg, seed=seed), batch_size, num_workers, seed)
    history, run, t0 = [], 0.0, time.time()
    net.train()
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(device), y.to(device)
        loss = loss_fn(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run += loss.item()
        if step % log_every == 0:
            rec = {"step": step, "loss": run / log_every, "sec": time.time() - t0}
            if val_dataset is not None:
                rec["val_loss"] = _val_loss(net, val_dataset, loss_fn, batch_size, val_batches, device)
            history.append(rec)
            msg = f"step {step:6d}  loss {rec['loss']:.4f}" + (f"  val {rec['val_loss']:.4f}" if "val_loss" in rec else "")
            print(msg + f"  {rec['sec']:6.0f}s", flush=True)
            if callback is not None:
                callback(history)
            run = 0.0
        if step >= steps:
            break
    net.eval()
    if out_path is not None:
        M.save(net, out_path)
    return net, history


def evaluate_dataset(predictor, ds: DiskDataset, n: int = None) -> dict:
    """저장된 이미지(보통 val split)로 재는 waypoint 별 오차. 증강 없이 원본을 쓴다."""
    n = len(ds) if n is None else min(n, len(ds))
    errs = []
    for i in range(n):
        pred = predictor.predict(ds.load_image(i))
        errs.append(np.hypot(*(pred - ds.wps[ds.idx[i]]).T))
    errs = np.array(errs)
    per = errs.mean(0)
    return {"per_waypoint_m": per, "mean_m": float(per.mean()), "max_m": float(errs.max()), "n": int(n)}
