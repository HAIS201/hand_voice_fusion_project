# -*- coding: utf-8 -*-
import csv
from pathlib import Path

import numpy as np
import librosa
from moviepy import VideoFileClip
from tqdm import tqdm

# 프로젝트 루트 경로
ROOT = Path(r"D:\pycharm\hand\fusion_project")
# 원본 비디오 데이터 폴더
DATA = ROOT / "data"
# 라벨 정보가 저장된 CSV 파일 경로
LABELS = ROOT / "labels.csv"

# 추출된 음성 특징을 저장할 폴더
OUTDIR = ROOT / "features" / "audio"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 오디오 샘플링 레이트
SR = 16000
# Mel 필터 개수
N_MELS = 64
# 시간 축 프레임 수를 60으로 고정
TARGET_T = 60


def load_audio_from_video(vpath: Path, sr: int = SR) -> np.ndarray:
    """
    동영상 파일에서 오디오 신호를 추출하는 함수
    반환값: 단일 채널(모노) 오디오 파형 (float32)
    """
    try:
        # 동영상 파일 열기
        with VideoFileClip(str(vpath)) as clip:
            # 오디오 트랙 존재 여부 확인
            if clip.audio is None:
                print(f"주의: {vpath.name} 에 오디오 트랙이 없습니다.")
                return np.array([], dtype=np.float32)

            # 오디오 신호를 샘플링 레이트(sr)로 추출
            arr = clip.audio.to_soundarray(fps=sr).astype(np.float32)

            # 스테레오 → 모노 변환 (두 채널 평균)
            if arr.ndim == 2 and arr.shape[1] > 1:
                y = arr.mean(axis=1)
            else:
                # 이미 모노이면 1차원으로 reshape
                y = arr.reshape(-1)

            return y

    except Exception as e:
        # 오디오 추출 실패 시 경고 출력
        print(f"경고: 오디오 추출 실패 {vpath.name} - {e}")
        return np.array([], dtype=np.float32)


def preprocess_audio(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    오디오 전처리 함수
    1. 앞뒤 무음 구간 제거
    2. 피크(최댓값) 기준 볼륨 정규화
    """
    # 빈 신호인 경우 그대로 반환
    if len(y) == 0:
        return y

    # 앞뒤 무음(trim) 제거 시도
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        # 모두 잘려 나간 경우 원본 신호 사용
        if len(y_trimmed) == 0:
            y_trimmed = y
    except Exception as e:
        print(f"오디오 trim 실패: {e}")
        y_trimmed = y

    # 피크 노멀라이제이션(최댓값이 1이 되도록 스케일 조정)
    max_val = np.max(np.abs(y_trimmed))
    if max_val > 0:
        y_normalized = y_trimmed / max_val
    else:
        y_normalized = y_trimmed

    return y_normalized


def mel_to_60(y: np.ndarray, sr: int = SR, n_mels: int = N_MELS, T: int = TARGET_T) -> np.ndarray:
    """
    Mel-spectrogram을 계산하고 시간 축을 T(60) 프레임으로 리샘플링
    반환값: (T, n_mels) 형태의 특징 행렬
    """
    # 빈 오디오인 경우 0 행렬 반환
    if y.size == 0:
        print("주의: 오디오가 비어 있어 0 행렬을 반환합니다.")
        return np.zeros((T, n_mels), dtype=np.float32)

    # 오디오 길이(초 단위) 계산
    duration = len(y) / sr
    # 너무 짧은 오디오는 사용하지 않고 0 행렬 반환
    if duration < 0.1:
        print(f"주의: 오디오 길이가 너무 짧습니다 ({duration:.2f}초)")
        return np.zeros((T, n_mels), dtype=np.float32)

    try:
        # 길이에 따라 FFT 및 hop_length 파라미터를 다르게 설정
        if duration < 1.0:
            n_fft = 512
            hop_length = int(0.01 * sr)  # 10ms 간격
        else:
            n_fft = 1024
            hop_length = int(0.02 * sr)  # 20ms 간격

        # Mel-spectrogram 계산
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=50,     # 최소 주파수
            fmax=8000,   # 최대 주파수
            power=2.0
        )

        # 전력을 dB 스케일로 변환
        mel_db = librosa.power_to_db(mel_spec, ref=np.max, amin=1e-10)

        # 현재 시간 프레임 수
        Tprime = mel_db.shape[1]
        if Tprime == 0:
            return np.zeros((T, n_mels), dtype=np.float32)

        # 시간 축을 T 개로 선형 리샘플링
        idxs = np.linspace(0, Tprime - 1, T, dtype=int)
        # (n_mels, Tprime) → 인덱싱 후 (T, n_mels)로 transpose
        mel_60 = mel_db[:, idxs].T

        return mel_60.astype(np.float32)

    except Exception as e:
        print(f"Mel-spectrogram 계산 실패: {e}")
        return np.zeros((T, n_mels), dtype=np.float32)


# labels.csv 읽기
rows = []
with open(LABELS, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 통계용 변수 초기화
processed_count = 0      # 성공적으로 특징을 추출한 샘플 수
failed_count = 0         # 처리 실패한 샘플 수
audio_durations = []     # 각 샘플의 오디오 길이(초)를 기록

print("음성 특징 추출을 시작합니다...")
for r in tqdm(rows, desc="오디오 특징 추출"):
    subset = r["subset"]

    # B_voice 와 C_fusion 샘플만 처리
    if subset not in ("B_voice", "C_fusion"):
        continue

    # 비디오 상대 경로 및 출력 npy 경로 설정
    rel = r["relpath"]
    inpath = DATA / rel
    outnpy = OUTDIR / (Path(rel).with_suffix(".npy").name)

    # 이미 처리된 파일이면 건너뜀
    if outnpy.exists():
        processed_count += 1
        continue

    # 입력 비디오 파일 존재 여부 확인
    if not inpath.exists():
        print(f"경고: 파일을 찾을 수 없습니다: {rel}")
        failed_count += 1
        continue

    try:
        # 동영상에서 오디오 신호 추출
        audio_wave = load_audio_from_video(inpath, sr=SR)

        # 오디오 길이(초 단위) 기록
        duration = len(audio_wave) / SR
        audio_durations.append(duration)

        # 오디오 전처리 (무음 제거 + 볼륨 정규화)
        audio_processed = preprocess_audio(audio_wave, sr=SR)

        # Mel-spectrogram 특징 계산 후 60프레임으로 리샘플링
        mel_features = mel_to_60(audio_processed, sr=SR, n_mels=N_MELS, T=TARGET_T)

        # 특징 행렬 크기 검증
        if mel_features.shape != (TARGET_T, N_MELS):
            print(f"경고: {rel} 특징 형태가 예상과 다릅니다: {mel_features.shape}")
            # 시간 축 프레임 수가 맞지 않을 경우 한 번 더 리샘플링 시도
            if mel_features.shape[0] != TARGET_T:
                mel_features = mel_to_60(audio_processed, sr=SR, n_mels=N_MELS, T=TARGET_T)

        # npy 파일로 저장
        outnpy.parent.mkdir(parents=True, exist_ok=True)
        np.save(outnpy, mel_features)
        processed_count += 1

    except Exception as e:
        # 처리 중 예외 발생 시 로그 출력 후 실패 카운트 증가
        print(f"오류: {rel} 처리 중 예외 발생 - {e}")
        failed_count += 1
        continue

# 최종 통계 정보 출력
print("\n" + "=" * 50)
print("음성 특징 추출이 완료되었습니다!")
print(f"CSV 전체 행 수(라벨 수): {len(rows)}")
print(f"성공적으로 처리된 샘플: {processed_count}")
print(f"처리에 실패한 샘플: {failed_count}")

if audio_durations:
    # 평균, 최소, 최대 길이 계산
    avg_duration = np.mean(audio_durations)
    min_duration = np.min(audio_durations)
    max_duration = np.max(audio_durations)

    print(f"평균 오디오 길이: {avg_duration:.2f}초")
    print(f"가장 짧은 오디오: {min_duration:.2f}초")
    print(f"가장 긴 오디오: {max_duration:.2f}초")

    # 0.5초보다 짧은 오디오 개수 집계
    short_audio = sum(1 for d in audio_durations if d < 0.5)
    if short_audio > 0:
        print(f"주의: 0.5초 미만 오디오가 {short_audio}개 있습니다.")

print(f"특징 파일 저장 경로: {OUTDIR}")
print("=" * 50)
