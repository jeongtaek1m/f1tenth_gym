import numpy as np, torch, pytest
from camsim import config, track, dataset, model, gt

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    return cfg, track.from_csv("examples/example_waypoints.csv", cfg)

def bev_hw(cfg):
    from camsim.render import bev_size
    return bev_size(cfg)

def test_make_sample_is_bev_with_camera_mask(ctx):
    """모델 입력은 BEV. 카메라가 못 보는 근거리(0.32 m 안쪽)는 바닥색이어야 함 (실차 IPM 출력과 동일)."""
    cfg, trk = ctx
    bev, wp, pose, cam = dataset.make_sample(trk, cfg, np.random.default_rng(0), with_camera=True)
    h, w = bev_hw(cfg)
    assert bev.shape == (h, w, 3) and cam.shape == (cfg.camera.image_height, cfg.camera.image_width, 3)
    assert wp.shape == (2,)
    near_rows = int((cfg.bev.x_range_m[1] - 0.30) / cfg.bev.resolution_m)      # x < 0.30 m -> 아래쪽 행들
    assert np.all(bev[near_rows:] == cfg.lane.color_floor)
    assert np.all(bev == cfg.lane.color_tape, axis=-1).sum() > 1000

def test_dataset_yields_tensor_pairs(ctx):
    cfg, trk = ctx
    x, y = next(iter(dataset.SynthDataset(trk, cfg, seed=0)))
    assert x.shape == (3, *bev_hw(cfg))
    assert x.dtype == torch.float32 and 0 <= x.min() <= x.max() <= 1
    assert y.shape == (2,)

def test_dataset_is_deterministic_per_seed(ctx):
    cfg, trk = ctx
    a = next(iter(dataset.SynthDataset(trk, cfg, seed=5)))
    b = next(iter(dataset.SynthDataset(trk, cfg, seed=5)))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

def test_dataloader_batches(ctx):
    cfg, trk = ctx
    dl = torch.utils.data.DataLoader(dataset.SynthDataset(trk, cfg), batch_size=4, num_workers=0)
    x, y = next(iter(dl))
    assert x.shape == (4, 3, *bev_hw(cfg)) and y.shape == (4, 2)

def test_model_forward_and_size(ctx):
    cfg, _ = ctx
    net = model.WaypointNet()
    assert net(torch.zeros(2, 3, *bev_hw(cfg))).shape == (2, 2)
    assert sum(p.numel() for p in net.parameters()) < 1_000_000

def test_predictor_shape(ctx):
    cfg, _ = ctx
    wp = model.Predictor(model.WaypointNet(), cfg).predict(np.zeros((*bev_hw(cfg), 3), np.uint8))
    assert wp.shape == (2,)

def test_oracle_matches_gt(ctx):
    cfg, trk = ctx
    pose = np.array([*trk.center[20], trk.heading[20]])
    o = model.OraclePredictor(trk, cfg)
    o.set_pose(pose)
    assert np.allclose(o.predict(None), gt.waypoint_ahead(pose, trk, cfg))

def test_save_load(ctx, tmp_path):
    cfg, _ = ctx
    net = model.WaypointNet()
    model.save(net, tmp_path / "m.pt", cfg)
    net2 = model.load(tmp_path / "m.pt", cfg)
    x = torch.zeros(1, 3, *bev_hw(cfg))
    net.eval(); net2.eval()
    assert torch.allclose(net(x), net2(x))

@pytest.mark.parametrize("change", ["bev", "waypoint", "color"])
def test_load_rejects_changed_input_spec(ctx, tmp_path, change):
    """다른 규격으로 학습한 체크포인트는 조용히 로드되면 안 됨."""
    from copy import deepcopy
    cfg, _ = ctx
    path = tmp_path / "m.pt"
    model.save(model.WaypointNet(), path, cfg)
    changed = deepcopy(cfg)
    if change == "bev": changed.bev.resolution_m *= 2
    elif change == "waypoint": changed.waypoints.ahead_m += 0.5
    else: changed.lane.color_tape = [0, 0, 255]
    with pytest.raises(ValueError, match="input_spec"):
        model.load(path, changed)

def test_predict_camera_matches_predict_on_ipm(ctx):
    """실차 경로(카메라 -> IPM -> predict)는 같은 BEV 를 직접 넣은 것과 같아야 함."""
    from camsim import camera, render
    cfg, trk = ctx
    H_g2i, H_i2g = camera.build(cfg)
    p = model.Predictor(model.WaypointNet(), cfg)
    cam = render.render(np.array([*trk.center[50], trk.heading[50]]), trk.quads, None, H_g2i, cfg)
    assert np.allclose(p.predict_camera(cam, H_i2g), p.predict(render.ipm_bev(cam, H_i2g, cfg)))
