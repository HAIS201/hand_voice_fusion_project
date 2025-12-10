# -*- coding: utf-8 -*-
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 프로젝트 루트 및 주요 경로 설정
ROOT = Path(r"D:\pycharm\hand\fusion_project")
SPLITS_DIR = ROOT / "splits"
FEAT_G_DIR = ROOT / "features" / "gesture"
FEAT_A_DIR = ROOT / "features" / "audio"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 사용 장치(CPU / GPU) 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------- utils -----------------
def seed_all(seed: int = 42):
    """난수 시드를 고정하여 재현성 확보"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_maps():
    """이동/행동 클래스 목록 및 클래스 → 인덱스 매핑 로드"""
    with open(SPLITS_DIR / "class_move.json", "r", encoding="utf-8") as f:
        moves = json.load(f)
    with open(SPLITS_DIR / "class_act.json", "r", encoding="utf-8") as f:
        acts = json.load(f)
    # 레이블 문자열을 0 ~ C-1 인덱스로 매핑
    m2i = {c: i for i, c in enumerate(moves)}
    a2i = {c: i for i, c in enumerate(acts)}
    return moves, acts, m2i, a2i


def rel_to_npy(relpath: str):
    """비디오 상대 경로(relpath)를 제스처/오디오 특징 npy 파일 경로로 변환"""
    rel = Path(relpath)
    name = rel.with_suffix(".npy").name
    return (FEAT_G_DIR / name, FEAT_A_DIR / name)


def safe_load_npy(path: Path, default_shape: tuple):
    """
    npy 파일을 안전하게 로드하는 함수
    - 파일이 없거나 형식/값이 이상하면 기본 0 행렬을 반환
    """
    if not path.exists():
        return np.zeros(default_shape, dtype=np.float32)

    try:
        data = np.load(path)

        # 2차원인지 확인 (T, D)
        if len(data.shape) != 2:
            print(f"경고: {path.name} 배열 차원 이상 {data.shape}")
            return np.zeros(default_shape, dtype=np.float32)

        # NaN 값 존재 여부 확인
        if np.isnan(data).any():
            print(f"경고: {path.name} 에 NaN 값이 포함되어 있습니다.")
            return np.zeros(default_shape, dtype=np.float32)

        return data.astype(np.float32)

    except Exception as e:
        print(f"경고: {path.name} 로드 실패 - {e}")
        return np.zeros(default_shape, dtype=np.float32)


# ----------------- Dataset -----------------
class EarlyFusionDataset(Dataset):
    """조기 융합(Early Fusion)용 제스처+오디오 데이터셋"""

    def __init__(self, csv_file: Path, m2i: dict, a2i: dict):
        # split CSV 로드
        self.df = pd.read_csv(csv_file)
        self.m2i = m2i  # move 레이블 → 인덱스
        self.a2i = a2i  # act 레이블 → 인덱스

        # 데이터셋 정보 출력
        print(f"데이터셋 로드: {csv_file.name}")
        print(f"  샘플 수: {len(self.df)}")

        # 제스처/오디오 특징 파일 존재 여부 점검
        missing_g = 0
        missing_a = 0
        for _, row in self.df.iterrows():
            g_path, a_path = rel_to_npy(row["relpath"])
            if not g_path.exists():
                missing_g += 1
            if not a_path.exists():
                missing_a += 1
        if missing_g > 0:
            print(f"  경고: 제스처 특징 파일이 없는 샘플 {missing_g}개")
        if missing_a > 0:
            print(f"  경고: 오디오 특징 파일이 없는 샘플 {missing_a}개")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 한 행(row) 읽기
        r = self.df.iloc[idx]
        g_path, a_path = rel_to_npy(r["relpath"])

        # np.load 대신 safe_load_npy 사용
        g = safe_load_npy(g_path, default_shape=(60, 63))
        a = safe_load_npy(a_path, default_shape=(60, 64))

        # 텍스트 레이블에서 인덱스 얻기
        move = r["move_label"]
        act = r["act_label"]
        # get()을 사용하여 없는 키에 대해 -1(무시할 라벨) 반환
        m_idx = self.m2i.get(move, -1)
        a_idx = self.a2i.get(act, -1)

        # 텐서 변환 및 기타 메타데이터 반환
        return {
            "g": torch.from_numpy(g).float(),  # (60, 63)
            "a": torch.from_numpy(a).float(),  # (60, 64)
            "mi": torch.tensor(m_idx, dtype=torch.long),
            "ai": torch.tensor(a_idx, dtype=torch.long),
            "subset": r["subset"],
            "env": r.get("env", "E?"),
            "move_label": move,
            "act_label": act,
            "relpath": r["relpath"],
        }


# ----------------- Model -----------------
class GRUEncoder(nn.Module):
    """단일 모달리티 시퀀스를 인코딩하는 GRU 인코더"""

    def __init__(self, input_dim, hidden=128, num_layers=1, bidir=True, drop=0.1):
        super().__init__()
        # GRU 기반 시퀀스 인코더
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        self.dropout = nn.Dropout(drop)
        # 양방향 여부에 따라 출력 차원 설정
        self.out_dim = hidden * (2 if bidir else 1)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.gru(x)  # (B, T, H*dir)
        # 마지막 타임스텝의 은닉 상태를 대표 벡터로 사용
        h = out[:, -1, :]  # (B, H*dir)
        return self.dropout(h)


class EarlyFusionGRU(nn.Module):
    """제스처 + 오디오 조기 융합 GRU 모델"""

    def __init__(self, num_move, num_act, drop=0.1):
        super().__init__()
        # 제스처 인코더 (입력 63차원)
        self.enc_g = GRUEncoder(63, hidden=128, bidir=True, drop=drop)
        # 오디오 인코더 (입력 64차원)
        self.enc_a = GRUEncoder(64, hidden=128, bidir=True, drop=drop)
        # 두 인코더의 출력을 concat 하므로 입력 차원 = 256 + 256 = 512
        fusion_in = self.enc_g.out_dim + self.enc_a.out_dim
        # 융합 후 비선형 변환 블록
        self.fuse = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, 128),   # 중간 차원을 128로 축소
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        # 최종 이동(move) 및 행동(act) 분류 헤드
        self.head_move = nn.Linear(128, num_move)
        self.head_act = nn.Linear(128, num_act)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Xavier 초기화 등으로 가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, g_seq, a_seq):
        """
        g_seq: (B, T, 63) 제스처 특징 시퀀스
        a_seq: (B, T, 64) 오디오 특징 시퀀스
        """
        hg = self.enc_g(g_seq)
        ha = self.enc_a(a_seq)
        # 인코더 출력 concat 후 융합 네트워크 통과
        h = self.fuse(torch.cat([hg, ha], dim=-1))
        # 이동/행동 각각에 대해 로짓 계산
        lm = self.head_move(h)
        la = self.head_act(h)
        return lm, la


# ----------------- metrics -----------------
def masked_ce_loss(logits, targets):
    """
    레이블이 -1인 샘플은 무시하는 CrossEntropy 손실
    (targets == -1 인 항목은 학습에 사용하지 않음)
    """
    mask = targets != -1
    if mask.sum() == 0:
        # 유효한 타겟이 하나도 없을 경우, 0 손실을 반환하되 그래디언트는 유지
        return torch.tensor(0., device=logits.device, requires_grad=True)
    return nn.functional.cross_entropy(logits[mask], targets[mask])


@torch.no_grad()
def eval_split(model, loader):
    """주어진 DataLoader에 대해 이동/행동 정확도를 계산"""
    model.eval()
    m_corr = m_total = 0
    a_corr = a_total = 0

    for batch in loader:
        g = batch["g"].to(DEVICE).float()
        a = batch["a"].to(DEVICE).float()
        mi = batch["mi"].to(DEVICE)
        ai = batch["ai"].to(DEVICE)

        lm, la = model(g, a)

        # 이동(move) 정확도 계산
        m_mask = mi != -1
        if m_mask.any():
            m_pred = lm.argmax(dim=-1)
            m_corr += (m_pred[m_mask] == mi[m_mask]).sum().item()
            m_total += m_mask.sum().item()

        # 행동(act) 정확도 계산
        a_mask = ai != -1
        if a_mask.any():
            a_pred = la.argmax(dim=-1)
            a_corr += (a_pred[a_mask] == ai[a_mask]).sum().item()
            a_total += a_mask.sum().item()

    m_acc = m_corr / m_total if m_total > 0 else 0.0
    a_acc = a_corr / a_total if a_total > 0 else 0.0
    return m_acc, a_acc


@torch.no_grad()
def eval_cfusion_csr(model, moves, acts):
    """
    C_fusion 테스트 세트에서 CSR(Command Success Rate) 계산
    - 이동 + 행동이 모두 정답일 때만 성공으로 간주
    """
    df = pd.read_csv(SPLITS_DIR / "test.csv")
    df = df[df["subset"] == "C_fusion"].reset_index(drop=True)

    if len(df) == 0:
        print("경고: C_fusion 테스트 세트가 비어 있습니다.")
        return 0.0, 0

    model.eval()
    ok = 0
    N = len(df)

    for _, r in df.iterrows():
        g_path, a_path = rel_to_npy(r["relpath"])

        # safe_load_npy 사용
        g = safe_load_npy(g_path, default_shape=(60, 63))
        a = safe_load_npy(a_path, default_shape=(60, 64))

        # 배치 차원 추가 후 텐서 변환
        g_tensor = torch.from_numpy(g).unsqueeze(0).to(DEVICE).float()
        a_tensor = torch.from_numpy(a).unsqueeze(0).to(DEVICE).float()

        # 추론
        lm, la = model(g_tensor, a_tensor)
        m_pred = moves[lm.argmax(dim=-1).item()]
        a_pred = acts[la.argmax(dim=-1).item()]

        # 이동 + 행동 모두 정답인지 확인
        if m_pred == r["move_label"] and a_pred == r["act_label"]:
            ok += 1

    csr = ok / N if N > 0 else 0.0
    return csr, N


@torch.no_grad()
def eval_cfusion_csr_by_env(model, moves, acts):
    """
    C_fusion 테스트 세트에서 환경(env)별 CSR 계산
    - 각 환경에 대해 이동+행동 모두 정답인 비율을 집계
    """
    df = pd.read_csv(SPLITS_DIR / "test.csv")
    df = df[df["subset"] == "C_fusion"].reset_index(drop=True)

    if len(df) == 0:
        return {}, 0

    model.eval()
    stats = {}  # env → {"ok": ..., "total": ...}

    for _, r in df.iterrows():
        env = r.get("env", "E?")
        if env not in stats:
            stats[env] = {"ok": 0, "total": 0}
        stats[env]["total"] += 1

        g_path, a_path = rel_to_npy(r["relpath"])
        g = safe_load_npy(g_path, default_shape=(60, 63))
        a = safe_load_npy(a_path, default_shape=(60, 64))

        g_tensor = torch.from_numpy(g).unsqueeze(0).to(DEVICE).float()
        a_tensor = torch.from_numpy(a).unsqueeze(0).to(DEVICE).float()

        lm, la = model(g_tensor, a_tensor)
        m_pred = moves[lm.argmax(dim=-1).item()]
        a_pred = acts[la.argmax(dim=-1).item()]

        if (m_pred == r["move_label"]) and (a_pred == r["act_label"]):
            stats[env]["ok"] += 1

    # 환경별 CSR 계산
    csr_env = {}
    for env, d in stats.items():
        csr_env[env] = d["ok"] / d["total"] if d["total"] > 0 else 0.0

    total = sum(d["total"] for d in stats.values())
    return csr_env, total


# ----------------- train -----------------
def train(args):
    """조기 융합 GRU 모델 학습 메인 함수"""
    # 시드 고정
    seed_all(args.seed)

    # 클래스 매핑 로드
    moves, acts, m2i, a2i = load_class_maps()
    num_move, num_act = len(moves), len(acts)

    print("=" * 80)
    print("조기 융합 (Early Fusion) GRU 모델 학습")
    print("=" * 80)
    print(f"이동(move) 클래스 수: {num_move} ({moves})")
    print(f"행동(act) 클래스 수: {num_act} ({acts})")

    # 데이터셋 생성
    print("\n데이터셋 로드 중...")
    train_ds = EarlyFusionDataset(SPLITS_DIR / "train.csv", m2i, a2i)
    val_ds = EarlyFusionDataset(SPLITS_DIR / "val.csv", m2i, a2i)
    test_ds = EarlyFusionDataset(SPLITS_DIR / "test.csv", m2i, a2i)

    # DataLoader 생성
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    print(f"\n데이터 로더 통계:")
    print(f"  학습(train) 세트: {len(train_ds)} 개 샘플")
    print(f"  검증(val) 세트: {len(val_ds)} 개 샘플")
    print(f"  테스트(test) 세트: {len(test_ds)} 개 샘플")

    # 모델 생성
    print("\n모델 생성 중...")
    model = EarlyFusionGRU(num_move=num_move, num_act=num_act, drop=args.drop).to(DEVICE)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 옵티마이저
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 학습률 스케줄러 (ReduceLROnPlateau 사용, verbose 옵션 제거)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=10
    )

    # 학습 루프 설정
    best_val = -1.0
    best_epoch = 0
    ckpt_path = OUT_DIR / "early_gru_best.pth"

    print(f"\n학습 시작 (총 {args.epochs} 에폭)")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            g = batch["g"].to(DEVICE).float()
            a = batch["a"].to(DEVICE).float()
            mi = batch["mi"].to(DEVICE)
            ai = batch["ai"].to(DEVICE)

            # 순전파
            lm, la = model(g, a)

            # 이동/행동 손실 계산
            loss_m = masked_ce_loss(lm, mi)
            loss_a = masked_ce_loss(la, ai)
            loss = loss_m + loss_a

            # 역전파
            opt.zero_grad()
            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()

            # 통계 값 누적
            bs = g.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            # 10 배치마다 중간 손실 출력
            if batch_idx % 10 == 0:
                print(f"  epoch {epoch} | batch {batch_idx}/{len(train_loader)} | loss: {loss.item():.4f}")

        # 에폭 종료 후 검증/테스트 정확도 평가
        val_m, val_a = eval_split(model, val_loader)
        test_m, test_a = eval_split(model, test_loader)

        dt = time.time() - t0
        tr_loss = running_loss / max(1, n_samples)
        val_score = (val_m + val_a) / 2.0

        # 검증 성능 기준으로 스케줄러 업데이트
        scheduler.step(val_score)

        # 에폭 로그 출력
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} | "
            f"val(m)={val_m:.3f} val(a)={val_a:.3f} val(avg)={val_score:.3f} | "
            f"test(m)={test_m:.3f} test(a)={test_a:.3f} | "
            f"lr={opt.param_groups[0]['lr']:.2e} | {dt:.1f}s"
        )

        # 최적(최고 val_score) 모델 저장
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_score": best_val,
                    "args": vars(args),
                    "moves": moves,
                    "acts": acts,
                },
                ckpt_path,
            )
            print(f"  -> 최적 모델을 {ckpt_path} 에 저장했습니다. (val_avg={best_val:.3f})")

    print(f"\n최적 모델: epoch {best_epoch}, val_avg={best_val:.3f}")
    print("-" * 80)

    # 최적 모델 로드 후 최종 평가
    print("최적 모델을 로드하여 최종 평가를 수행합니다...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 테스트 집합 정확도 및 CSR 평가
    test_m_acc, test_a_acc = eval_split(model, test_loader)
    csr, N = eval_cfusion_csr(model, moves, acts)
    csr_env, _ = eval_cfusion_csr_by_env(model, moves, acts)

    print("\n" + "=" * 80)
    print("조기 융합 모델 최종 평가 결과")
    print("=" * 80)
    print(f"테스트 세트 - 이동 정확도: {test_m_acc:.3f}")
    print(f"테스트 세트 - 행동 정확도: {test_a_acc:.3f}")
    print(f"C_fusion 테스트 세트 CSR: {csr:.3f} (총 {N}개 샘플)")

    if csr_env:
        print("\n환경(env)별 C_fusion CSR:")
        for env in sorted(csr_env.keys()):
            print(f"  {env}: {csr_env[env]:.3f}")

    # 텍스트 보고서 저장
    with open(OUT_DIR / "early_gru_report.txt", "w", encoding="utf-8") as f:
        f.write("조기 융합(Early Fusion) GRU 학습 보고서\n")
        f.write("=" * 50 + "\n")
        f.write("학습 하이퍼파라미터:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n데이터셋 정보:\n")
        f.write(f"  학습 세트: {len(train_ds)} 개 샘플\n")
        f.write(f"  검증 세트: {len(val_ds)} 개 샘플\n")
        f.write(f"  테스트 세트: {len(test_ds)} 개 샘플\n")
        f.write("\n평가 결과:\n")
        f.write(f"  최고 검증 정확도 (avg): {best_val:.4f} (epoch {best_epoch})\n")
        f.write(f"  테스트 세트 이동 정확도: {test_m_acc:.4f}\n")
        f.write(f"  테스트 세트 행동 정확도: {test_a_acc:.4f}\n")
        f.write(f"  C_fusion 테스트 세트 CSR: {csr:.4f} ({N} clips)\n")
        if csr_env:
            f.write("\n환경(env)별 C_fusion CSR:\n")
            for env, v in sorted(csr_env.items()):
                f.write(f"  {env}: {v:.4f}\n")

    print(f"\n보고서가 {OUT_DIR / 'early_gru_report.txt'} 에 저장되었습니다.")
    print("조기 융합 모델 학습이 완료되었습니다!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="조기 융합 GRU 모델 학습 스크립트")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--bs", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률(learning rate)")
    parser.add_argument("--wd", type=float, default=1e-4, help="가중치 감쇠(weight decay)")
    parser.add_argument("--drop", type=float, default=0.3, help="Dropout 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 값")

    args = parser.parse_args()

    print("사용 장치:", DEVICE)
    print("조기 융합 모델 설정:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    train(args)
