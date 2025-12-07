import argparse
import json
import time
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- 경로 및 상수 ---------- #
ROOT       = Path(r"D:\pycharm\hand\fusion_project")
SPLITS_DIR = ROOT / "splits"
FEAT_G_DIR = ROOT / "features" / "gesture"
FEAT_A_DIR = ROOT / "features" / "audio"
OUT_DIR    = ROOT / "outputs"

EARLY_CKPT = OUT_DIR / "early_gru_best.pth"
LATE_CKPT  = OUT_DIR / "late_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unity UDP 주소 (다른 스크립트와 동일 포트 사용)
UNITY_ADDR = ("127.0.0.1", 5052)

DG = 63   # 제스처 feature 차원 (21*3)
DA = 64   # 오디오 feature 차원
TARGET_T = 60


# ---------- 유틸 함수 ---------- #
def seed_all(seed: int = 42):
    """파이썬 / 넘파이 / 파이토치 시드 고정"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_maps():
    """class_gesture.json / class_voice.json 에서 클래스 리스트 로드"""
    with open(SPLITS_DIR / "class_gesture.json", "r", encoding="utf-8") as f:
        gest = json.load(f)
    with open(SPLITS_DIR / "class_voice.json", "r", encoding="utf-8") as f:
        voic = json.load(f)
    g2i = {c: i for i, c in enumerate(gest)}
    v2i = {c: i for i, c in enumerate(voic)}
    return gest, voic, g2i, v2i


def rel_to_npy(relpath: str):
    """data 밑 mp4 relpath → features 밑 같은 이름의 npy 경로로 매핑"""
    rel = Path(relpath)
    npyname = rel.with_suffix(".npy").name
    return (FEAT_G_DIR / npyname, FEAT_A_DIR / npyname)


# ---------- Dataset: 시퀀스 그대로 읽기 ---------- #
class FusionTestSeqDataset(Dataset):
    """
    테스트용 Dataset (GRU 버전):

      - 시간 축을 유지한 채로 반환:
          g: (T, DG)
          a: (T, DA)
      - 문자열 라벨, subset, env 등의 정보도 함께 반환
    """
    def __init__(self, csv_file: Path):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def _to_seq(self, arr: np.ndarray, d_expect: int) -> np.ndarray:
        """
        다양한 형태의 npy를 (T, D_EXPECT) 형태로 통일:

          - [T,21,3] → reshape → [T,63]
          - [T,D]    → 그대로 사용
          - [D]      → T=1 로 간주
        """
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.squeeze(arr)

        if arr.ndim == 3:
            T = arr.shape[0]
            arr = arr.reshape(T, -1)
        elif arr.ndim == 2:
            # 이미 (T,D) 형태
            pass
        elif arr.ndim == 1:
            # (D,) → (1,D)
            arr = arr.reshape(1, -1)
        else:
            # 이상한 경우: 0으로 채운 한 프레임
            arr = np.zeros((1, d_expect), dtype=np.float32)

        # 마지막 차원이 기대한 차원과 다르면 최대한 맞춰줌
        if arr.shape[1] != d_expect:
            T = arr.shape[0]
            arr = arr.reshape(T, -1)
            if arr.shape[1] < d_expect:
                pad = np.zeros((T, d_expect - arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=1)
            elif arr.shape[1] > d_expect:
                arr = arr[:, :d_expect]

        return arr.astype(np.float32)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        g_path, a_path = rel_to_npy(r["relpath"])

        # 제스처 시퀀스
        if g_path.exists():
            g = np.load(g_path)
            g_seq = self._to_seq(g, DG)
        else:
            g_seq = np.zeros((TARGET_T, DG), dtype=np.float32)

        # 음성 시퀀스
        if a_path.exists():
            a = np.load(a_path)
            a_seq = self._to_seq(a, DA)
        else:
            a_seq = np.zeros((TARGET_T, DA), dtype=np.float32)

        return {
            "g": torch.from_numpy(g_seq),    # (T,DG)
            "a": torch.from_numpy(a_seq),    # (T,DA)
            "g_label": str(r["gesture_label"]).upper(),
            "v_label": str(r["voice_label"]).upper(),
            "subset": r["subset"],
            "env":    r["env"],
            "relpath": r["relpath"],
        }


# ---------- GRU Encoder & 모델 구조 ---------- #
class GRUEncoder(nn.Module):
    """입력 (B,T,D) → 출력 (B,H*dir) 로 바꿔주는 Bi-GRU 인코더"""
    def __init__(self, input_dim, hidden_dim=128,
                 num_layers=1, bidirectional=True, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

    @property
    def output_dim(self):
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def forward(self, x):
        # x: (B,T,D)
        out, _ = self.gru(x)
        # time-average pooling
        h_mean = out.mean(dim=1)  # (B,H*dir)
        return h_mean


class EarlyFusionGRU(nn.Module):
    """
    Early Fusion GRU:

      제스처 GRU + 음성 GRU → concat → MLP → gesture / voice 두 개의 head
    """
    def __init__(self, dg=DG, da=DA,
                 enc_hidden=128, enc_layers=1, drop=0.1,
                 fusion_dim=256, num_g=6, num_v=2):
        super().__init__()
        self.enc_g = GRUEncoder(
            dg, hidden_dim=enc_hidden,
            num_layers=enc_layers,
            bidirectional=True,
            dropout=drop,
        )
        self.enc_a = GRUEncoder(
            da, hidden_dim=enc_hidden,
            num_layers=enc_layers,
            bidirectional=True,
            dropout=drop,
        )
        fuse_in = self.enc_g.output_dim + self.enc_a.output_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.head_g = nn.Linear(fusion_dim, num_g)
        self.head_v = nn.Linear(fusion_dim, num_v)

    def forward(self, g_seq, a_seq):
        eg = self.enc_g(g_seq)  # (B,Hg)
        ea = self.enc_a(a_seq)  # (B,Ha)
        z  = torch.cat([eg, ea], dim=-1)
        h  = self.fuse(z)
        lg = self.head_g(h)
        lv = self.head_v(h)
        return lg, lv


class LateFusionGRU(nn.Module):
    """
    Late Fusion GRU (단일 모델 버전):

      - 제스처 GRU → 제스처 head
      - 음성   GRU → 음성 head
    """
    def __init__(self, dg=DG, da=DA,
                 enc_hidden=128, enc_layers=1, drop=0.1,
                 num_g=6, num_v=2):
        super().__init__()
        self.enc_g = GRUEncoder(
            dg, hidden_dim=enc_hidden,
            num_layers=enc_layers,
            bidirectional=True,
            dropout=drop,
        )
        self.enc_a = GRUEncoder(
            da, hidden_dim=enc_hidden,
            num_layers=enc_layers,
            bidirectional=True,
            dropout=drop,
        )

        self.head_g = nn.Linear(self.enc_g.output_dim, num_g)
        self.head_v = nn.Linear(self.enc_a.output_dim, num_v)

    def forward(self, g_seq, a_seq):
        eg = self.enc_g(g_seq)
        ea = self.enc_a(a_seq)
        lg = self.head_g(eg)
        lv = self.head_v(ea)
        return lg, lv


class GestureGRU(nn.Module):
    """손 제스처만 분류하는 GRU 브랜치 (ckpt['gnet'] / ckpt['g_model'] 등에 대응)"""
    def __init__(self, dg=DG, enc_hidden=128, enc_layers=1,
                 drop=0.1, num_g=6):
        super().__init__()
        self.enc = GRUEncoder(
            dg, hidden_dim=enc_hidden,
            num_layers=enc_layers,
            bidirectional=True,
            dropout=drop,
        )
        self.head = nn.Linear(self.enc.output_dim, num_g)

    def forward(self, g_seq):  # (B,T,DG)
        h = self.enc(g_seq)
        return self.head(h)     # (B,num_g)


class VoiceGRU(nn.Module):
    """음성만 분류하는 GRU 브랜치 (ckpt['vnet'] / ckpt['v_model'] 등에 대응)"""
    def __init__(self, da=DA, enc_hidden=128, enc_layers=1,
                 drop=0.1, num_v=2):
        super().__init__()
        self.enc = GRUEncoder(
            da, hidden_dim=enc_hidden,
            num_layers=enc_layers,
            bidirectional=True,
            dropout=drop,
        )
        self.head = nn.Linear(self.enc.output_dim, num_v)

    def forward(self, a_seq):  # (B,T,DA)
        h = self.enc(a_seq)
        return self.head(h)     # (B,num_v)


class TwoBranchLateWrapper(nn.Module):
    """
    Late Fusion 체크포인트가 g 브랜치 / v 브랜치로 따로 저장된 경우를 위한 래퍼.

    - forward(g_seq, a_seq) 에서 내부적으로
      g_model(g_seq), v_model(a_seq) 를 호출해서
      (logits_g, logits_v)를 반환한다. (LateFusionGRU 와 인터페이스 동일)
    """
    def __init__(self, g_model: GestureGRU, v_model: VoiceGRU):
        super().__init__()
        self.g_model = g_model
        self.v_model = v_model

    def forward(self, g_seq, a_seq):
        lg = self.g_model(g_seq)
        lv = self.v_model(a_seq)
        return lg, lv


# ---------- 체크포인트 로드 ---------- #
def load_early_model(gest, voic):
    """Early Fusion GRU 모델 로드"""
    ckpt = torch.load(EARLY_CKPT, map_location=DEVICE)
    num_g, num_v = len(gest), len(voic)
    model = EarlyFusionGRU(
        dg=DG, da=DA,
        enc_hidden=128, enc_layers=1,
        drop=0.1, fusion_dim=256,
        num_g=num_g, num_v=num_v
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[OK] EARLY 모델 로드 완료: {EARLY_CKPT}")
    return model


def load_late_model(gest, voic):
    """
    Late Fusion 체크포인트 로드.

    가능한 케이스:
      1) ckpt['model'] 에 LateFusionGRU 전체가 저장된 경우
      2) ckpt['g_model'], ckpt['v_model'] 로 분리 저장된 경우
      3) ckpt['gnet'], ckpt['vnet'] 이름으로 저장된 경우
    """
    ckpt = torch.load(LATE_CKPT, map_location=DEVICE)
    print("[INFO] late ckpt keys:", list(ckpt.keys()))

    num_g, num_v = len(gest), len(voic)

    # case 1: 단일 "model" 키
    if "model" in ckpt:
        model = LateFusionGRU(
            dg=DG, da=DA,
            enc_hidden=128, enc_layers=1,
            drop=0.1, num_g=num_g, num_v=num_v
        ).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"[OK] LATE 모델 로드 완료 (single 'model'): {LATE_CKPT}")
        return model

    # case 2: "g_model" / "v_model" 쌍
    elif "g_model" in ckpt and "v_model" in ckpt:
        g_state = ckpt["g_model"]
        v_state = ckpt["v_model"]

        g_model = GestureGRU(
            dg=DG, enc_hidden=128, enc_layers=1,
            drop=0.1, num_g=num_g
        ).to(DEVICE)
        v_model = VoiceGRU(
            da=DA, enc_hidden=128, enc_layers=1,
            drop=0.1, num_v=num_v
        ).to(DEVICE)

        g_model.load_state_dict(g_state)
        v_model.load_state_dict(v_state)
        g_model.eval()
        v_model.eval()
        print(f"[OK] LATE g_model & v_model 로드 완료: {LATE_CKPT}")

        wrapper = TwoBranchLateWrapper(g_model, v_model).to(DEVICE)
        wrapper.eval()
        return wrapper

    # case 3: "gnet" / "vnet" 쌍 (지금 사용하는 형태)
    elif "gnet" in ckpt and "vnet" in ckpt:
        g_state = ckpt["gnet"]
        v_state = ckpt["vnet"]

        g_model = GestureGRU(
            dg=DG, enc_hidden=128, enc_layers=1,
            drop=0.1, num_g=num_g
        ).to(DEVICE)
        v_model = VoiceGRU(
            da=DA, enc_hidden=128, enc_layers=1,
            drop=0.1, num_v=num_v
        ).to(DEVICE)

        g_model.load_state_dict(g_state)
        v_model.load_state_dict(v_state)
        g_model.eval()
        v_model.eval()
        print(f"[OK] LATE gnet & vnet 로드 완료: {LATE_CKPT}")

        wrapper = TwoBranchLateWrapper(g_model, v_model).to(DEVICE)
        wrapper.eval()
        return wrapper

    # case 4: 위 세 가지 어디에도 해당 안 되는 경우
    else:
        raise KeyError(
            "late_best.pth 안에 'model' / 'g_model'+'v_model' / 'gnet'+'vnet' "
            f"형태가 없습니다. 실제 keys={list(ckpt.keys())}"
        )


# ---------- Unity로 명령 보내기 ---------- #
def send_to_unity(sock, g_cmd: str, v_cmd: str):
    """
    g_cmd / v_cmd 예시:
      - 'FORWARD' / 'BACKWARD' / 'LEFT' / 'RIGHT'
      - 'ATTACK' / 'DEFEND' / 'NONE'
    """
    if g_cmd and g_cmd != "NONE":
        msg = f"G:{g_cmd}"
        sock.sendto(msg.encode("utf-8"), UNITY_ADDR)
        print("[SEND][G]", g_cmd)
    if v_cmd and v_cmd != "NONE":
        msg = f"V:{v_cmd}"
        sock.sendto(msg.encode("utf-8"), UNITY_ADDR)
        print("[SEND][V]", v_cmd)


# ---------- 메인 로직: A_hand / B_voice 만 재생 ---------- #
def main(args):
    seed_all(args.seed)
    gest, voic, g2i, v2i = load_class_maps()

    full_ds = FusionTestSeqDataset(SPLITS_DIR / "test.csv")

    # subset 기준으로 필터링: A_hand / B_voice 만 남기고 C_fusion 은 완전히 제거
    if args.subset == "A_hand":
        df = full_ds.df[full_ds.df["subset"] == "A_hand"].reset_index(drop=True)
    elif args.subset == "B_voice":
        df = full_ds.df[full_ds.df["subset"] == "B_voice"].reset_index(drop=True)
    else:  # both
        df = full_ds.df[full_ds.df["subset"].isin(["A_hand", "B_voice"])].reset_index(drop=True)

    ds = FusionTestSeqDataset(SPLITS_DIR / "test.csv")
    # 위에서 필터링한 df 를 그대로 덮어써서 사용
    ds.df = df

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    if args.mode == "early":
        early_model = load_early_model(gest, voic)
        late_model  = None
    else:
        early_model = None
        late_model  = load_late_model(gest, voic)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[INFO] Unity UDP target = {UNITY_ADDR}")
    print(f"[INFO] mode={args.mode}, subset={args.subset}, delay={args.delay}s")

    idx = 0
    for batch in loader:
        subset = batch["subset"][0]  # A_hand 또는 B_voice
        env    = batch["env"][0]
        rel    = batch["relpath"][0]
        g_gt   = batch["g_label"][0]
        v_gt   = batch["v_label"][0]

        idx += 1

        g_seq = batch["g"].to(DEVICE).float()   # (B,T,DG)
        a_seq = batch["a"].to(DEVICE).float()   # (B,T,DA)

        # 추론
        if args.mode == "early":
            with torch.no_grad():
                lg, lv = early_model(g_seq, a_seq)
        else:
            with torch.no_grad():
                lg, lv = late_model(g_seq, a_seq)

        gi_pred = lg.argmax(dim=-1).item()
        vi_pred = lv.argmax(dim=-1).item()

        g_pred = gest[gi_pred] if len(gest) > 0 else "NONE"
        v_pred = voic[vi_pred] if len(voic) > 0 else "NONE"

        print("\n------------------------------------------------")
        print(f"[{idx}] subset={subset}  env={env}  rel={rel}")
        print(f"  GT  : G={g_gt} , V={v_gt}")
        print(f"  PRED: G={g_pred} , V={v_pred}")

        # A_hand: 제스처만 전송, B_voice: 음성만 전송
        if subset == "A_hand":
            g_cmd = g_pred
            v_cmd = "NONE"
        else:  # B_voice
            g_cmd = "NONE"
            v_cmd = v_pred

        send_to_unity(sock, g_cmd, v_cmd)

        time.sleep(args.delay)

    print("\n[DONE] 재생이 끝났습니다. subset / mode 를 바꿔서 다시 돌려봐도 됩니다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        default="early",
        choices=["early", "late"],
        help="어떤 모델을 사용할지 선택 (early_gru_best / late_best)",
    )
    ap.add_argument(
        "--subset",
        type=str,
        default="both",
        choices=["A_hand", "B_voice", "both"],
        help="어떤 subset 을 재생할지 선택 (C_fusion 은 포함되지 않음)",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="각 샘플 사이 대기 시간 (초 단위)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
