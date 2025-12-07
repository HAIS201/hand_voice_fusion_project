import argparse
import json
import random
from pathlib import Path

import pandas as pd

# 프로젝트 루트 및 경로 설정
ROOT = Path(r"D:\pycharm\hand\fusion_project")
LABELS = ROOT / "labels.csv"
OUTDIR = ROOT / "splits"
OUTDIR.mkdir(parents=True, exist_ok=True)


def normalize_paths(df: pd.DataFrame) -> pd.DataFrame:
    """윈도우 경로의 역슬래시를 슬래시(/)로 통일"""
    df = df.copy()
    df["relpath"] = df["relpath"].astype(str).str.replace("\\", "/", regex=False)
    return df


def add_dual_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    단일 label 컬럼을 (gesture_label, voice_label) 두 컬럼으로 분리해서 저장
    - A_hand : 제스처 라벨만 사용, 음성은 NONE
    - B_voice: 음성 라벨만 사용, 제스처는 NONE
    - C_fusion: "MOVE+ACTION" 형태를 파싱해서 둘 다 사용
    """
    df = df.copy()
    g_list, v_list = [], []

    for subset, label in zip(df["subset"], df["label"]):
        up = str(label).upper().strip()

        if subset == "A_hand":
            g_list.append(up)
            v_list.append("NONE")

        elif subset == "B_voice":
            g_list.append("NONE")
            v_list.append(up)

        elif subset == "C_fusion":
            if "+" in up:
                g, v = up.split("+", 1)
                g_list.append(g.strip())
                v_list.append(v.strip())
            else:
                # 예외적으로 '+'가 없으면 둘 다 NONE 처리
                g_list.append("NONE")
                v_list.append("NONE")

        else:
            g_list.append("NONE")
            v_list.append("NONE")

    df["gesture_label"] = g_list
    df["voice_label"] = v_list
    return df


def stratified_split_by_env(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    subset 별로 (label, env) 조합을 기준으로 층화 분할을 수행.
    장점:
    - 같은 클래스라도 환경(env)별 샘플이 train/val/test에 골고루 분포되도록 함.
    """
    rng = random.Random(seed)
    parts = []

    for subset in sorted(df["subset"].unique()):
        sub_df = df[df["subset"] == subset].copy()
        sub_df["split"] = ""  # train / val / test

        # (label, env) 별로 그룹화
        grouped = sub_df.groupby(["label", "env"])
        for (lbl, env), grp in grouped:
            idx = grp.index.tolist()
            rng.shuffle(idx)
            n = len(idx)

            # 샘플 수가 적어도 가능한 한 train/val/test로 나누려고 시도
            if n >= 3:
                n_test = max(1, int(round(n * test_ratio)))
                n_val = max(1, int(round(n * val_ratio)))
            else:
                # 샘플이 너무 적을 때는 비율 기반을 조금 약하게 적용
                n_test = max(0, int(round(n * test_ratio)))
                n_val = max(0, int(round(n * val_ratio)))

            if n_test + n_val > n:
                # 극단적으로 적은 케이스에서는 train 위주로 두고
                # 필요하면 test만 조금 남기는 방향으로 조정
                n_test = min(1, n)
                n_val = 0

            n_train = n - n_val - n_test

            test_idx = idx[:n_test]
            val_idx = idx[n_test:n_test + n_val]
            train_idx = idx[n_test + n_val:]

            if n_train > 0:
                sub_df.loc[train_idx, "split"] = "train"
            if n_val > 0:
                sub_df.loc[val_idx, "split"] = "val"
            if n_test > 0:
                sub_df.loc[test_idx, "split"] = "test"

        parts.append(sub_df)

    full = pd.concat(parts, ignore_index=True)
    return full


def save_class_lists(df: pd.DataFrame, outdir: Path):
    """
    제스처/음성 클래스 목록을 JSON으로 저장
    - "NONE"은 제외
    """
    gest = sorted([c for c in df["gesture_label"].unique() if c != "NONE"])
    voic = sorted([c for c in df["voice_label"].unique() if c != "NONE"])

    with open(outdir / "class_gesture.json", "w", encoding="utf-8") as f:
        json.dump(gest, f, ensure_ascii=False, indent=2)

    with open(outdir / "class_voice.json", "w", encoding="utf-8") as f:
        json.dump(voic, f, ensure_ascii=False, indent=2)

    print("[OK] class_gesture.json:", gest)
    print("[OK] class_voice.json:", voic)


def main(seed: int, val_ratio: float, test_ratio: float):
    # labels.csv 읽기
    df = pd.read_csv(LABELS)

    need_cols = {"subset", "relpath", "label", "env"}
    miss = need_cols - set(df.columns)
    if miss:
        raise SystemExit(f"[ERROR] labels.csv에 필요한 컬럼이 없습니다: {sorted(miss)}")

    df = normalize_paths(df)
    df = add_dual_labels(df)

    # 층화 분할 실행
    split_df = stratified_split_by_env(df, val_ratio, test_ratio, seed)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(OUTDIR / "full_with_splits.csv", index=False, encoding="utf-8")
    split_df[split_df["split"] == "train"].to_csv(OUTDIR / "train.csv", index=False, encoding="utf-8")
    split_df[split_df["split"] == "val"].to_csv(OUTDIR / "val.csv", index=False, encoding="utf-8")
    split_df[split_df["split"] == "test"].to_csv(OUTDIR / "test.csv", index=False, encoding="utf-8")

    print(f"[OK] train/val/test 분할 결과를 {OUTDIR} 에 저장했습니다.")
    save_class_lists(df, OUTDIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    ap.add_argument("--val", type=float, default=0.2, help="validation 비율")
    ap.add_argument("--test", type=float, default=0.2, help="test 비율")
    args = ap.parse_args()

    main(seed=args.seed, val_ratio=args.val, test_ratio=args.test)