import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
import csv
import math

# 프로젝트 루트 경로 (실제 폴더 구조에 맞게 설정)
ROOT = Path(r"D:\pycharm\hand\fusion_project")
DATA = ROOT / "data"
LABELS = ROOT / "labels.csv"

# 제스처용 피처(.npy)를 저장할 폴더
OUTDIR = ROOT / "features" / "gesture"
OUTDIR.mkdir(parents=True, exist_ok=True)

# MediaPipe Hands 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # 영상 스트림이므로 False
    max_num_hands=2,               # 최대 2개 손 검출
    min_detection_confidence=0.5,  # 손 검출 최소 신뢰도
    min_tracking_confidence=0.5    # 추적 최소 신뢰도
)

# 모든 시퀀스를 60프레임으로 통일 (부족하면 복제/0패딩, 많으면 균등 샘플링)
TARGET_T = 60


def sample_to_T(seq, T=TARGET_T):
    """
    seq: [N, 21, 3]
    -> 길이가 N인 프레임 시퀀스를 선형적으로 T 프레임으로 리샘플링
    """
    if len(seq) == 0:
        return np.zeros((T, 21, 3), dtype=np.float32)
    idxs = np.linspace(0, len(seq) - 1, T)
    out = np.stack([seq[int(round(i))] for i in idxs], axis=0)
    return out.astype(np.float32)


def normalize_hand(frame):
    """
    frame: [21, 3]
    -> 손목(랜드마크 0)을 원점으로 평행이동,
       손바닥 폭(랜드마크 5~17 거리)으로 xy를 정규화
    """
    wrist = frame[0, :2]
    palm = np.linalg.norm(frame[5, :2] - frame[17, :2]) + 1e-6
    xy = (frame[:, :2] - wrist) / palm
    z = frame[:, 2:3]
    return np.concatenate([xy, z], axis=1)


# labels.csv 읽어서 메타 정보 로드
rows = []
with open(LABELS, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# 각 비디오에 대해 제스처 피처 추출
for r in tqdm(rows, desc="gesture"):
    rel = r["relpath"]
    inpath = DATA / rel
    outnpy = OUTDIR / (Path(rel).with_suffix(".npy").name)

    cap = cv2.VideoCapture(str(inpath))
    seq = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # BGR -> RGB (MediaPipe는 RGB를 기대)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(frame)

        # 여기서는 간단히 "탐지된 첫 번째 손"만 그 구간의 대표 손으로 사용
        # (필요하면 r["subset"]에 따라 Right/Left를 나눠 처리 가능)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]  # 첫 번째 손
            h, w, _ = frame.shape
            pts = []
            for p in lm.landmark:
                pts.append([p.x * w, p.y * h, p.z])
            pts = np.array(pts, dtype=np.float32)
            pts = normalize_hand(pts)  # [21, 3]
            seq.append(pts)
        else:
            # 손이 검출되지 않으면 이전 프레임을 복제하거나, 없으면 0으로 채움
            if len(seq) > 0:
                seq.append(seq[-1])
            else:
                seq.append(np.zeros((21, 3), dtype=np.float32))

    cap.release()

    # [T, 21, 3] 형태로 통일
    seq = np.stack(seq, axis=0) if len(seq) > 0 else np.zeros((1, 21, 3), np.float32)
    seq = sample_to_T(seq, TARGET_T)  # [60, 21, 3]

    # .npy로 저장
    np.save(outnpy, seq)