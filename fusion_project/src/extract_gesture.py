# -*- coding: utf-8 -*-
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# 프로젝트 루트 경로
ROOT = Path(r"D:\pycharm\hand\fusion_project")
# 원본 비디오 데이터 폴더
DATA = ROOT / "data"
# 라벨 정보가 저장된 CSV 파일 경로
LABELS = ROOT / "labels.csv"

# 추출된 손 제스처 특징을 저장할 폴더
OUTDIR = ROOT / "features" / "gesture"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 모든 시퀀스를 60프레임으로 고정
TARGET_T = 60

# MediaPipe Hands 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # 동영상(연속 프레임) 처리 모드
    max_num_hands=2,               # 최대 인식 손 개수
    min_detection_confidence=0.5,  # 손 검출 신뢰도 임계값
    min_tracking_confidence=0.5,   # 손 추적 신뢰도 임계값
)


def normalize_frame(pts: np.ndarray) -> np.ndarray:
    """
    pts: (21, 3), 각 랜드마크의 (x, y) 픽셀 좌표와 z 값
    - 손목(랜드마크 0)을 기준으로 평행이동
    - 손바닥 너비(랜드마크 5와 17 사이 거리)로 정규화
    반환: (21, 3)을 평탄화한 63차원 벡터
    """
    # 손목(0번 랜드마크)의 (x, y) 좌표
    wrist = pts[0, :2]
    # 손바닥 너비(5번과 17번 랜드마크 사이 거리)에 작은 값 더해 0으로 나누기 방지
    palm = np.linalg.norm(pts[5, :2] - pts[17, :2]) + 1e-6
    # 손목 기준 평행이동 후 손바닥 너비로 스케일 정규화
    xy = (pts[:, :2] - wrist) / palm
    # z 값은 그대로 사용
    z = pts[:, 2:3]
    # (21, 3) 형태로 다시 결합
    normed = np.concatenate([xy, z], axis=1)
    # (63,) 벡터로 평탄화하여 반환
    return normed.reshape(-1)


def resample_seq(seq: np.ndarray, T: int = TARGET_T) -> np.ndarray:
    """
    seq: (N, D) 형태의 시퀀스를 선형 인덱스 기반으로 (T, D)로 리샘플링
    N: 원본 프레임 수, D: 특징 차원(여기서는 63)
    """
    # 프레임이 하나도 없을 경우 0으로 채운 시퀀스 반환
    if len(seq) == 0:
        return np.zeros((T, 63), dtype=np.float32)

    # 안전성 체크: 2차원이며 두 번째 차원이 63인지 확인
    if seq.ndim != 2 or seq.shape[1] != 63:
        print(f"경고: 시퀀스 형태 이상 {seq.shape}, 0으로 초기화합니다.")
        return np.zeros((T, 63), dtype=np.float32)

    # 0 ~ N-1 구간을 T개 지점으로 균일하게 나누어 인덱스 생성
    idxs = np.linspace(0, len(seq) - 1, T)
    # 가장 가까운 프레임을 선택하여 새로운 시퀀스 구성
    out = np.stack([seq[int(round(i))] for i in idxs], axis=0)
    return out.astype(np.float32)


def extract_hand_features(frame, w, h):
    """
    한 프레임에서 손 랜드마크를 검출하고,
    정규화된 63차원 손 특징 벡터를 추출한다.
    반환:
      - feat: (63,) 특징 벡터
      - detected: 손 검출 여부(True/False)
    """
    # OpenCV BGR → MediaPipe용 RGB 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        # 첫 번째로 검출된 손만 사용
        lm = res.multi_hand_landmarks[0]
        pts = []
        # 각 랜드마크의 (x, y, z) 값을 픽셀 좌표 기준으로 수집
        for p in lm.landmark:
            pts.append([p.x * w, p.y * h, p.z])
        pts = np.array(pts, dtype=np.float32)
        # 한 프레임을 63차원 벡터로 정규화
        feat = normalize_frame(pts)
        return feat, True
    else:
        # 손이 검출되지 않은 경우 0 벡터 반환
        return np.zeros((63,), dtype=np.float32), False


# labels.csv 읽기
rows = []
with open(LABELS, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 통계 정보 초기화
processed_count = 0  # 성공적으로 처리된 샘플 수
failed_count = 0     # 실패한 샘플 수
detection_stats = []  # 각 비디오별 손 검출률 리스트

print("손 제스처 특징 추출을 시작합니다...")
for r in tqdm(rows, desc="제스처 특징 추출"):
    subset = r["subset"]

    # A_hand 와 C_fusion 샘플만 처리
    if subset not in ("A_hand", "C_fusion"):
        continue

    # 비디오 상대 경로 및 출력 npy 경로 설정
    rel = r["relpath"]
    inpath = DATA / rel
    outnpy = OUTDIR / (Path(rel).with_suffix(".npy").name)

    # 이미 처리된 경우 건너뜀
    if outnpy.exists():
        processed_count += 1
        continue

    # 입력 비디오 파일 존재 여부 확인
    if not inpath.exists():
        print(f"경고: 파일을 찾을 수 없습니다: {rel}")
        failed_count += 1
        continue

    try:
        # 비디오 캡처 열기
        cap = cv2.VideoCapture(str(inpath))
        if not cap.isOpened():
            print(f"경고: 비디오를 열 수 없습니다: {rel}")
            failed_count += 1
            continue

        seq = []             # 각 프레임의 손 특징을 저장할 리스트
        detected_frames = 0  # 손이 검출된 프레임 수
        total_frames = 0     # 전체 프레임 수

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1
            h, w, _ = frame.shape
            feat, detected = extract_hand_features(frame, w, h)

            if detected:
                detected_frames += 1

            seq.append(feat)

        cap.release()

        # 비디오 단위 손 검출률 계산
        detection_rate = detected_frames / total_frames if total_frames > 0 else 0
        detection_stats.append(detection_rate)

        # 손 검출률이 너무 낮은 경우 경고 출력
        if detection_rate < 0.3 and total_frames > 10:
            print(f"주의: {rel} 손 검출률이 낮습니다 ({detection_rate:.1%})")

        # 프레임이 하나도 없는 경우, 0 벡터 한 개로 대체
        if len(seq) == 0:
            seq = [np.zeros((63,), dtype=np.float32)]

        # 리스트를 (N, 63) 배열로 변환 후, (TARGET_T, 63)으로 리샘플링
        seq_array = np.stack(seq, axis=0)          # (N, 63)
        seq_resampled = resample_seq(seq_array, TARGET_T)  # (60, 63)

        # npy 파일로 저장
        outnpy.parent.mkdir(parents=True, exist_ok=True)
        np.save(outnpy, seq_resampled)
        processed_count += 1

    except Exception as e:
        # 예외 발생 시 로그 출력 후 실패 카운트 증가
        print(f"오류: {rel} 처리 중 예외 발생 - {e}")
        failed_count += 1
        continue

# MediaPipe 리소스 해제
hands.close()

# 최종 통계 정보 출력
print("\n" + "=" * 50)
print("손 제스처 특징 추출이 완료되었습니다!")
print(f"CSV 전체 행 수(라벨 수): {len(rows)}")
print(f"성공적으로 처리된 샘플: {processed_count}")
print(f"처리에 실패한 샘플: {failed_count}")

if detection_stats:
    avg_detection = np.mean(detection_stats)
    min_detection = np.min(detection_stats)
    max_detection = np.max(detection_stats)
    print(f"평균 손 검출률: {avg_detection:.1%}")
    print(f"최저 손 검출률: {min_detection:.1%}")
    print(f"최고 손 검출률: {max_detection:.1%}")

    # 검출률이 30% 미만인 샘플 개수 집계
    low_detect = sum(1 for rate in detection_stats if rate < 0.3)
    if low_detect > 0:
        print(f"경고: 손 검출률이 30% 미만인 샘플이 {low_detect}개 있습니다.")

print(f"특징 저장 경로: {OUTDIR}")
print("=" * 50)
