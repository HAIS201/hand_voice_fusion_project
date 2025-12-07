import numpy as np
import librosa
from moviepy import VideoFileClip
from pathlib import Path
from tqdm import tqdm
import csv

# 기본 경로 설정
ROOT   = Path(r"D:\pycharm\hand\fusion_project")
DATA   = ROOT / "data"
LABELS = ROOT / "labels.csv"

# 오디오 피처(.npy)를 저장할 폴더
OUTDIR = ROOT / "features" / "audio"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 오디오 파라미터
SR        = 16000   # 타겟 샘플링 레이트
N_MELS    = 64      # Mel 필터 뱅크 개수
TARGET_T  = 60      # 제스처 60프레임과 맞추기 (시간 스텝 60으로 통일)


def load_audio_from_video(vpath: Path, sr: int = SR) -> np.ndarray:
    """
    MoviePy 2.x를 사용해서 MP4 파일에서 오디오를 바로 numpy 배열로 추출.
    - 단일 채널(mono) + 지정한 sr로 리샘플링
    - 오디오 트랙이 없거나 실패하면 길이 0짜리 배열 반환
    """
    try:
        # with 블록으로 열고 자동으로 리소스 정리
        with VideoFileClip(str(vpath)) as clip:
            if clip.audio is None:
                return np.array([], dtype=np.float32)

            # 지정한 샘플링 레이트로 오디오 배열 추출 (shape: [N, channels])
            arr = clip.audio.to_soundarray(fps=sr).astype(np.float32)

            # 스테레오인 경우 두 채널 평균을 내서 모노로 변환
            if arr.ndim == 2 and arr.shape[1] > 1:
                y = arr.mean(axis=1)
            else:
                y = arr.reshape(-1)

            return y

    except Exception as e:
        print(f"[WARN] 오디오 읽기 실패: {vpath.name} -> {e.__class__.__name__}")
        return np.array([], dtype=np.float32)


def mel_to_60(
    y: np.ndarray,
    sr: int = SR,
    n_mels: int = N_MELS,
    target_t: int = TARGET_T
) -> np.ndarray:
    """
    1) 입력 파형 y 로부터 Mel-spectrogram 계산
    2) 시간 축을 균등 리샘플링해서 (target_t, n_mels) 형태로 맞추기
       -> 최종 출력 shape: [target_t, n_mels], dtype=float32
    """
    if y.size == 0:
        return np.zeros((target_t, n_mels), dtype=np.float32)

    # Mel 스펙트로그램 계산 (파워 스펙트럼 기준)
    m = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=int(0.02 * sr)  # 약 20 ms 단위로 프레임 나누기
    )

    # dB 스케일로 변환 (로그 스케일)
    m = librosa.power_to_db(m, ref=np.max)  # [n_mels, T']

    Tprime = m.shape[1]
    # 시간 축(T')을 target_t 개로 균등 샘플링
    idxs = np.linspace(0, Tprime - 1, target_t)
    m60 = np.stack([m[:, int(round(i))] for i in idxs], axis=0)  # [target_t, n_mels]

    return m60.astype(np.float32)


# labels.csv 읽기
rows = []
with open(LABELS, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 각 비디오 파일에 대해 오디오 피처 추출
for r in tqdm(rows, desc="audio"):
    rel = r["relpath"]
    inpath = DATA / rel
    outnpy = OUTDIR / (Path(rel).with_suffix(".npy").name)

    # 비디오에서 오디오 파형 추출
    y = load_audio_from_video(inpath, sr=SR)

    # Mel-spectrogram -> [60, N_MELS]로 변환
    a = mel_to_60(y, sr=SR, n_mels=N_MELS, target_t=TARGET_T)

    # 상위 폴더가 없으면 생성
    outnpy.parent.mkdir(parents=True, exist_ok=True)

    # .npy로 저장
    np.save(outnpy, a)