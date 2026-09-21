# camsim — 합성 카메라 waypoint 파이프라인

`f1tenth_gym` 은 카메라를 못 그린다. 그런데 바닥 테이프 트랙은 전부 평면이라, 캘리브레이션 행렬
하나만 있으면 "이 위치에서 카메라로 보면 이렇게 보인다"를 계산해서 가짜 영상을 그릴 수 있다.
이 패키지가 그 렌더러와 학습, gym 폐루프 검증을 담는다.

## 실행: 노트북 하나

학생용 경로는 `notebooks/camsim_lab.ipynb` 하나고, 코랩에서 돌리는 걸 기준으로 만들었다.

0장 설치·파라미터 → 1장 카메라와 트랙 → 2장 GT 와 데이터셋 → 3장 학습(train/val loss 실시간)
→ 4장 폐루프(스윕 표, 횡오차 곡선, 주행 영상) → 5장 결과 보관.

각 장 첫 셀의 파라미터를 바꾸고 그 장을 다시 실행하면서 뭐가 달라지는지 본다.
`config.yaml` 은 기본값이고, 노트북 파라미터 셀이 그 위에 덮어쓴다.

노트북 원본은 `camsim/scripts/build_notebook.py` 다. 노트북을 고칠 일이 있으면 그 파일을 고치고
다시 생성한다.

    python camsim/scripts/build_notebook.py

## 학생이 손대는 파일

`camsim/config.yaml` 하나. 카메라 높이·각도·화각, 테이프 폭, waypoint 거리, 지연, 속도가 전부 거기 있다.
캘리브레이션이 끝나면 `camera.h_i2g_file` 에 `.npy` 경로를 적는다. 그러면 카메라 가정값은 무시된다.

## 모델 입력은 BEV

앞에서 본 영상이 아니라 위에서 내려다본 BEV 를 모델에 넣는다. 이유는 실차와 규격을 맞추기 위해서다.

    시뮬 : render.render_bev(pose, quads, cfg, mask) 가 테이프를 top-down 으로 바로 그린다
    실차 : 카메라 -> undistort -> render.ipm_bev(cam, H_i2g, cfg)

`mask = render.bev_visibility_mask(H_g2i, cfg)` 로 카메라가 못 보는 영역(코앞 사각지대, 화각 밖)을
바닥색으로 덮으면 실차 IPM 출력과 같은 모양이 된다. 그래서 `Predictor.predict(bev)` 하나를 양쪽이
공유할 수 있다. 실차용 편의 함수로 `Predictor.predict_camera(cam, H_i2g)` 도 있다.

원근 렌더(`render.render`)는 눈으로 확인하거나 IPM 과 비교할 때만 쓴다.
BEV 범위와 해상도는 `config.yaml` 의 `bev:` 섹션 하나로 정한다.

## 데이터셋

`dataset.generate_dataset` 이 `out/dataset/images/NNNNNN.png` (BEV) 와 `labels.csv`
(file, x, y, theta, wp0_x..wp5_y) 를 만든다. 저장되는 건 증강 없는 원본이고, train/val 은 9:1 로
결정적으로 나뉜다.

증강은 로딩 때 `DiskDataset(..., augment_fn=fn)` 의 `fn(bev, rng) -> bev` 하나로 넣는다 (기본 None).
같은 데이터로 증강만 바꿔 가며 비교할 수 있게 이렇게 나눠 놨다.

`camsim/augment.py` 에 OpenCV 기반 증강이 들어 있다. 세기는 config 의 `augment:` 에서 오고 기본값은
전부 "변화 없음"이다.

| 분류 | 함수 |
|---|---|
| 기하 | `jitter_bev` (pitch 변화로 BEV 가 휨), `ipm_blur` (먼 곳일수록 뭉개짐), `erase_patches` (테이프 마모) |
| 조명 | `brightness_contrast`, `gamma`, `hsv_shift`, `illumination`, `shadow` |
| 센서 | `blur`, `noise`, `jpeg` |
| 조합 | `example_augment` |

`jitter_bev` 외에는 실제 카메라 물리의 근사라, 실차 영상을 찍어 본 뒤 다시 설계하는 게 맞다.
`train.evaluate(..., degrade_fn=fn)` 으로 열화된 입력에 대한 강건성을 잴 수 있다.
sim-to-real 증강 설계가 노트북 3장 과제다. 실차 IPM 결과를 같은 `labels.csv` 포맷으로 저장하면
그대로 학습된다.

데이터는 코랩 로컬 디스크에 만들고 드라이브에는 zip 하나로 옮길 것. 드라이브는 작은 파일이 많으면
견딜 수 없이 느리다.

## 정답 경로 기준 (`waypoints.line`)

- `center` — 좌우 테이프의 중간선. 실차 HSV+IPM 자동 라벨링과 같은 기준이라 기본값이다.
- `racing` — `examples/example_waypoints.csv` 의 레이싱 라인. 코너 안쪽을 파고들어 더 짧다 (156 m vs 163 m).

어느 쪽이든 테이프(모델이 보는 것)는 똑같고 정답만 달라진다. 두 기준으로 학습해서 랩타임과 안정성을
비교하는 게 노트북 2장 실습이다. `Track.left_m` / `right_m` 이 그 기준선에서 좌우 테이프까지의
거리이고, pose 샘플링과 실격 판정이 이 값을 쓴다.

## 테이프 배치

`lane.follow_walls: true` 면 맵 PNG 의 벽까지 거리를 재서 `wall_margin_m` 안쪽에 테이프를 놓는다.
트랙 모양이 맵과 일치하는 대신 폭이 구간마다 다르다. `false` 면 중심선에서 `track_width_m/2` 로
일정하게 놓는다. 실습실 테이프 트랙과 같은 방식이라, 실차 트랙 치수가 정해지면 이쪽으로 바꾼다.

## 실격 규칙

실차 트랙은 벽 없이 테이프가 경계다. 시뮬도 같은 규칙을 쓴다. 차체
(`closed_loop.car_length_m` x `car_width_m`) 네 모서리 중 하나라도 테이프 안쪽 선을 넘으면
`reason="tape_crossed"` 로 종료한다. 완주(`lap`)는 테이프를 한 번도 안 넘고 한 바퀴 돈 것이다.
gym 벽 충돌(`collision`)은 이 맵에서 벽이 멀어서 거의 안 난다.

## 좌표계

world = gym 맵 (m). vehicle = 후륜축 원점, x 전방, y 좌측. image = OpenCV (u 우, v 아래).
`H_g2i` 는 ground (x,y,1) -> image. pitch 가 0이면 지평선이 이미지 세로 중앙에 온다.

## 코랩 주의

- 코랩 Python(3.13)에는 numpy 1.22 wheel 이 없다. 그래서 numpy 는 코랩 기본(2.x) 그대로 두고
  f110_gym 을 `pip install --no-deps -e .` 로 설치한다. 노트북 첫 셀이 이 순서대로 되어 있다.
- 설치 중에 numpy 가 바뀌면 런타임을 한 번 재시작해야 한다. 안 하면 `numpy.dtype size changed` 가 난다.
- 주행 영상이 셀에 안 뜨면 코덱 문제다. OpenCV 의 mp4v 는 브라우저가 못 읽으므로 `viz.to_h264()` 로
  변환해서 띄운다 (노트북은 이미 그렇게 한다).
- `pyglet` import 에러가 나면 `!apt-get install -y libgl1` 후 재시도. `f110_env` 가 모듈 최상단에서
  `from pyglet import gl` 을 해서, 창을 안 띄워도 GL 라이브러리가 있어야 한다.
- 노트북 첫 코드 셀의 `REPO_URL` 기본값은 조교 fork 다. 다른 fork 를 쓰면 그 줄만 바꾸면 된다.
  clone 이 실패하면 `%cd f1tenth_gym` 부터 전부 깨지므로 주소를 먼저 확인할 것.
- 세션이 끊기면 다 날아간다. 데이터 zip 과 `model.pt` 는 5장에서 드라이브로 옮겨 둘 것.

## 로컬에서 테스트만 돌리려면

gym 없이도 도는 테스트가 대부분이라, 렌더러나 트랙 쪽을 고쳤을 때는 로컬에서 바로 확인할 수 있다.

    pip install opencv-python-headless pyyaml pillow pytest
    python -m pytest camsim/tests -q --ignore=camsim/tests/test_closed_loop.py \
        --ignore=camsim/tests/test_dataset_model.py --ignore=camsim/tests/test_disk_dataset.py \
        --ignore=camsim/tests/test_train.py

전부 돌리려면 torch 와 f110_gym 이 필요하다. f110_gym 은 Python 3.9 + numpy 1.22 를 요구하고,
gym 0.19 는 setup.py 에 오타가 있어서 `camsim/scripts/install_gym019.sh` 로 따로 설치해야 한다.
그냥 코랩에서 돌리는 게 빠르다.
