import os, json, argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT       = Path(r"D:\pycharm\hand\fusion_project")
SPLITS_DIR = ROOT / "splits"
FEAT_G_DIR = ROOT / "features" / "gesture"
FEAT_A_DIR = ROOT / "features" / "audio"
OUT_DIR    = ROOT / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_T = 60   # 시퀀스 길이 (이미 60프레임으로 맞춰져 있음)
DG = 63        # gesture feature dim = 21*3
DA = 64        # audio feature dim = 64 mels


# ---------------- 공용 유틸 ---------------- #
def seed_all(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_maps():
    """gesture / voice 클래스 리스트 로딩"""
    with open(SPLITS_DIR / "class_gesture.json", "r", encoding="utf-8") as f:
        gest = json.load(f)
    with open(SPLITS_DIR / "class_voice.json", "r", encoding="utf-8") as f:
        voic = json.load(f)
    g2i = {c: i for i, c in enumerate(gest)}
    v2i = {c: i for i, c in enumerate(voic)}
    return gest, voic, g2i, v2i


def rel_to_npy(relpath: str):
    """data/ 밑 mp4 경로를 features/ 의 npy 경로로 매핑"""
    rel = Path(relpath)
    npyname = rel.with_suffix(".npy").name
    return (FEAT_G_DIR / npyname, FEAT_A_DIR / npyname)


# ---------------- 데이터 증강 ---------------- #
def augment_gesture_seq(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 21, 3) 가정
    간단한 노이즈 + 스케일 + 타임 쉬프트
    """
    x = seq.copy().astype(np.float32)
    if x.ndim != 3 or x.shape[1] != 21 or x.shape[2] != 3:
        return x

    # 가우시안 노이즈
    noise = np.random.normal(0.0, 0.02, size=x.shape).astype(np.float32)
    x += noise

    # 약간 확대/축소
    scale = np.random.uniform(0.9, 1.1)
    x[:, :, 0:2] *= scale

    # 시간 축 이동
    shift = np.random.randint(-3, 4)
    if shift != 0:
        x = np.roll(x, shift=shift, axis=0)

    return x


def spec_augment(mel: np.ndarray,
                 max_time_mask: int = 6,
                 max_freq_mask: int = 8) -> np.ndarray:
    """
    mel: (T, F) 가정
    간단한 SpecAugment + 전체 gain
    """
    x = mel.copy().astype(np.float32)
    if x.ndim != 2:
        return x

    T, F = x.shape

    # 시간 마스크
    t_mask = np.random.randint(0, max_time_mask + 1)
    if t_mask > 0 and T > t_mask:
        t0 = np.random.randint(0, T - t_mask)
        x[t0:t0 + t_mask, :] = x.mean()

    # 주파수 마스크
    f_mask = np.random.randint(0, max_freq_mask + 1)
    if f_mask > 0 and F > f_mask:
        f0 = np.random.randint(0, F - f_mask)
        x[:, f0:f0 + f_mask] = x.mean()

    # 전체 gain
    gain = np.random.uniform(-1.0, 1.0)
    x += gain

    return x


# ---------------- Dataset (시퀀스 유지) ---------------- #
class SeqFusionDataset(Dataset):
    """
    splits/train.csv / val.csv / test.csv 를 그대로 사용
    - gesture: (T,21,3) -> (T,63) 시퀀스로 사용
    - audio  : (T,64)   -> (T,64) 그대로 사용
    """
    def __init__(self, csv_file: Path, g2i: dict, v2i: dict,
                 split_name: str = "train", augment: bool = True):
        self.df = pd.read_csv(csv_file)
        self.g2i = g2i
        self.v2i = v2i
        self.split_name = split_name
        self.use_augment = augment and (split_name == "train")

    def __len__(self):
        return len(self.df)

    def _load_gesture_seq(self, path: Path) -> np.ndarray:
        if path.exists():
            arr = np.load(path)
        else:
            # 누락 시 0 시퀀스로 채움
            return np.zeros((TARGET_T, DG), dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32)
        # 보통 (60,21,3) 으로 저장되어 있음
        if arr.ndim == 3:
            T, J, C = arr.shape
            arr = arr.reshape(T, -1)  # (T,63)
        elif arr.ndim == 2:
            # 이미 (T, D) 형태
            pass
        else:
            # 이상한 경우는 0으로
            arr = np.zeros((TARGET_T, DG), dtype=np.float32)

        # 길이가 맞지 않으면 간단히 cut/pad
        T, D = arr.shape
        if T > TARGET_T:
            arr = arr[:TARGET_T]
        elif T < TARGET_T:
            pad = np.zeros((TARGET_T - T, D), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        return arr

    def _load_audio_seq(self, path: Path) -> np.ndarray:
        if path.exists():
            arr = np.load(path)
        else:
            return np.zeros((TARGET_T, DA), dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32)
        # 보통 (T,64)
        if arr.ndim == 2:
            pass
        elif arr.ndim == 1:
            # (F,) -> (T,F) 로 복제
            arr = np.tile(arr[None, :], (TARGET_T, 1))
        else:
            # 이상한 경우 0
            arr = np.zeros((TARGET_T, DA), dtype=np.float32)

        T, F = arr.shape
        if T > TARGET_T:
            arr = arr[:TARGET_T]
        elif T < TARGET_T:
            pad = np.zeros((TARGET_T - T, F), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        return arr

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        g_path, a_path = rel_to_npy(r["relpath"])

        g_seq = self._load_gesture_seq(g_path)  # (T,DG)
        a_seq = self._load_audio_seq(a_path)    # (T,DA)

        if self.use_augment:
            # 제스처는 (T,21,3)로 원복 후 증강 -> 다시 (T,63)
            g_raw = g_seq.reshape(TARGET_T, 21, 3)
            g_raw = augment_gesture_seq(g_raw)
            g_seq = g_raw.reshape(TARGET_T, -1)
            a_seq = spec_augment(a_seq)

        # 라벨
        g_label = str(r["gesture_label"]).upper()
        v_label = str(r["voice_label"]).upper()
        g_idx = self.g2i[g_label] if g_label in self.g2i else -1
        v_idx = self.v2i[v_label] if v_label in self.v2i else -1

        return {
            "g": torch.from_numpy(g_seq),   # (T,DG)
            "a": torch.from_numpy(a_seq),   # (T,DA)
            "gi": torch.tensor(g_idx, dtype=torch.long),
            "vi": torch.tensor(v_idx, dtype=torch.long),
            "subset": r["subset"],
            "label":  r["label"],
            "env":    r["env"],
            "relpath": r["relpath"],
        }


# ---------------- 모델 (GRU 인코더 + Early Fusion) ---------------- #
class GRUEncoder(nn.Module):
    """
    시퀀스 인코더: 입력 (B,T,input_dim) -> 출력 (B,enc_dim)
    Bi-GRU + time-average pooling
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=1,
                 bidirectional=True, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

    @property
    def output_dim(self):
        return self.hidden_dim * (2 if self.bidirectional else 1)

    def forward(self, x):
        """
        x: (B,T,input_dim)
        returns: (B, output_dim)  (time-average pooling)
        """
        out, _ = self.gru(x)      # out: (B,T,H*dir)
        h_mean = out.mean(dim=1)  # (B,H*dir)
        return h_mean


class EarlyFusionGRUNet(nn.Module):
    """
    - 제스처 시퀀스 (B,T,63) -> GRU -> g_vec (B,Eg)
    - 음성   시퀀스 (B,T,64) -> GRU -> a_vec (B,Ea)
    - concat [g_vec,a_vec] -> fusion -> 두 개의 head (gesture / voice)
    """
    def __init__(self, g_input=DG, a_input=DA,
                 enc_hidden=128, enc_layers=1,
                 fusion_hidden=256, drop=0.1,
                 num_g=6, num_v=2):
        super().__init__()
        self.enc_g = GRUEncoder(g_input, hidden_dim=enc_hidden,
                                num_layers=enc_layers,
                                bidirectional=True, dropout=drop)
        self.enc_a = GRUEncoder(a_input, hidden_dim=enc_hidden,
                                num_layers=enc_layers,
                                bidirectional=True, dropout=drop)
        enc_dim = self.enc_g.output_dim  # == self.enc_a.output_dim

        self.fuse = nn.Sequential(
            nn.Linear(enc_dim * 2, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.head_g = nn.Linear(fusion_hidden, num_g)
        self.head_v = nn.Linear(fusion_hidden, num_v)

    def forward(self, g_seq, a_seq):
        """
        g_seq: (B,T,DG)
        a_seq: (B,T,DA)
        """
        eg = self.enc_g(g_seq)  # (B,Eg)
        ea = self.enc_a(a_seq)  # (B,Ea)
        z  = torch.cat([eg, ea], dim=-1)  # (B,Eg+Ea)
        h  = self.fuse(z)                 # (B,fusion_hidden)
        lg = self.head_g(h)
        lv = self.head_v(h)
        return lg, lv


# ---------------- Loss & Eval ---------------- #
def masked_ce_loss(logits, targets):
    """
    targets == -1 인 샘플은 무시하고 cross-entropy 계산
    """
    mask = targets != -1
    if mask.sum() == 0:
        return torch.tensor(0., device=logits.device, requires_grad=True)
    return nn.functional.cross_entropy(logits[mask], targets[mask])


@torch.no_grad()
def eval_split(model, loader):
    model.eval()
    g_correct = g_total = 0
    v_correct = v_total = 0

    for batch in loader:
        g = batch["g"].to(DEVICE).float()   # (B,T,DG)
        a = batch["a"].to(DEVICE).float()   # (B,T,DA)
        gi = batch["gi"].to(DEVICE)
        vi = batch["vi"].to(DEVICE)

        lg, lv = model(g, a)

        # gesture
        mask_g = gi != -1
        if mask_g.any():
            pred_g = lg.argmax(dim=-1)
            g_correct += (pred_g[mask_g] == gi[mask_g]).sum().item()
            g_total   += mask_g.sum().item()

        # voice
        mask_v = vi != -1
        if mask_v.any():
            pred_v = lv.argmax(dim=-1)
            v_correct += (pred_v[mask_v] == vi[mask_v]).sum().item()
            v_total   += mask_v.sum().item()

    g_acc = g_correct / g_total if g_total > 0 else 0.0
    v_acc = v_correct / v_total if v_total > 0 else 0.0
    return g_acc, v_acc, g_total, v_total


@torch.no_grad()
def eval_cfusion_pair(model, loader):
    """
    C_fusion 샘플에 대해서만:
    - gesture / voice 둘 다 맞으면 1, 아니면 0
    -> pair-accuracy 계산
    """
    model.eval()
    pair_correct = pair_total = 0

    for batch in loader:
        subset_list = list(batch["subset"])
        mask_c = torch.tensor(
            [s == "C_fusion" for s in subset_list],
            dtype=torch.bool
        )
        if not mask_c.any():
            continue

        g = batch["g"][mask_c].to(DEVICE).float()
        a = batch["a"][mask_c].to(DEVICE).float()
        gi = batch["gi"][mask_c].to(DEVICE)
        vi = batch["vi"][mask_c].to(DEVICE)

        lg, lv = model(g, a)
        pred_g = lg.argmax(dim=-1)
        pred_v = lv.argmax(dim=-1)

        ok = (pred_g == gi) & (pred_v == vi)
        pair_correct += ok.sum().item()
        pair_total   += ok.numel()

    pair_acc = pair_correct / pair_total if pair_total > 0 else 0.0
    return pair_acc, pair_total


# ---------------- Train Loop ---------------- #
def train(args):
    seed_all(args.seed)
    gest, voic, g2i, v2i = load_class_maps()
    num_g, num_v = len(gest), len(voic)

    train_ds = SeqFusionDataset(SPLITS_DIR / "train.csv", g2i, v2i,
                                split_name="train", augment=True)
    val_ds   = SeqFusionDataset(SPLITS_DIR / "val.csv",   g2i, v2i,
                                split_name="val", augment=False)
    test_ds  = SeqFusionDataset(SPLITS_DIR / "test.csv",  g2i, v2i,
                                split_name="test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                               num_workers=0)

    model = EarlyFusionGRUNet(
        g_input=DG, a_input=DA,
        enc_hidden=args.enc_hidden, enc_layers=args.enc_layers,
        fusion_hidden=args.fusion, drop=args.drop,
        num_g=num_g, num_v=num_v
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.wd)

    best_val = -1.0
    best_ckpt = OUT_DIR / "early_gru_best.pth"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_n    = 0

        for batch in train_loader:
            g = batch["g"].to(DEVICE).float()
            a = batch["a"].to(DEVICE).float()
            gi = batch["gi"].to(DEVICE)
            vi = batch["vi"].to(DEVICE)

            lg, lv = model(g, a)
            loss_g = masked_ce_loss(lg, gi)
            loss_v = masked_ce_loss(lv, vi)
            loss = loss_g + loss_v

            opt.zero_grad()
            loss.backward()
            opt.step()

            run_loss += loss.item() * g.size(0)
            run_n    += g.size(0)

        train_loss = run_loss / max(1, run_n)
        g_val, v_val, _, _ = eval_split(model, val_loader)
        g_test, v_test, _, _ = eval_split(model, test_loader)
        dt = time.time() - t0

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} | "
              f"val(g)={g_val:.3f} val(v)={v_val:.3f} | "
              f"test(g)={g_test:.3f} test(v)={v_test:.3f} | "
              f"{dt:.1f}s")

        val_score = (g_val + v_val) / 2.0
        if val_score > best_val:
            best_val = val_score
            torch.save({
                "model": model.state_dict(),
                "gest": gest, "voic": voic,
            }, best_ckpt)
            print(f"  -> saved best to {best_ckpt} (val_avg={best_val:.3f})")

    # best 모델로 최종 테스트 + pair-Acc 평가
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    g_acc, v_acc, gN, vN = eval_split(model, test_loader)
    pair_acc, pairN      = eval_cfusion_pair(model, test_loader)

    print("\n==== FINAL (best on val, Early-Fusion GRU) ====")
    print(f"TEST Gesture Acc: {g_acc:.3f} ({gN} items)")
    print(f"TEST Voice   Acc: {v_acc:.3f} ({vN} items)")
    print(f"TEST C_fusion Pair-Acc: {pair_acc:.3f} ({pairN} pairs)")

    report_path = OUT_DIR / "early_gru_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"TEST Gesture Acc: {g_acc:.4f} ({gN} items)\n")
        f.write(f"TEST Voice   Acc: {v_acc:.4f} ({vN} items)\n")
        f.write(f"TEST C_fusion Pair-Acc: {pair_acc:.4f} ({pairN} pairs)\n")
    print(f"[OK] report saved: {report_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--enc_hidden", type=int, default=128)
    ap.add_argument("--enc_layers", type=int, default=1)
    ap.add_argument("--fusion", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.1)
    args = ap.parse_args()
    train(args)
