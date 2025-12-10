# -*- coding: utf-8 -*-
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# 프로젝트 루트 및 경로 설정
ROOT = Path(r"D:\pycharm\hand\fusion_project")
LABELS = ROOT / "labels.csv"
SPLITS_DIR = ROOT / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def create_composite_label(row):
    """계층적(분층) 샘플링을 위한 복합 레이블 생성 함수"""
    move = row["move_label"] if pd.notna(row["move_label"]) else "NONE"
    act = row["act_label"] if pd.notna(row["act_label"]) else "NONE"
    return f"{move}+{act}"


def make_stratified_splits(df, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    분층(계층) 분할을 수행하여,
    각 subset 및 (move_label + act_label) 조합의 비율을 최대한 유지하도록
    train / val / test로 나누는 함수
    """
    # 랜덤 시드 고정
    np.random.seed(seed)
    random.seed(seed)

    # 복합 레이블 컬럼 추가
    df = df.copy()
    df["composite_label"] = df.apply(create_composite_label, axis=1)

    # 기본값은 모두 train 으로 설정
    df["split"] = "train"

    # 전체 데이터 분포 출력
    print("데이터 분포 통계:")
    print("-" * 40)
    print("subset 기준 통계:")
    print(df["subset"].value_counts())
    print("\n이동(move) 레이블 통계:")
    print(df["move_label"].value_counts())
    print("\n행동(act) 레이블 통계:")
    print(df["act_label"].value_counts())
    print("\n환경(env) 통계:")
    print(df["env"].value_counts())
    print("-" * 40)

    # subset 별로 나누어 처리
    for subset in df["subset"].unique():
        subset_df = df[df["subset"] == subset].copy()
        print(f"\nsubset {subset} 처리 중 (총 {len(subset_df)}개 샘플)")

        # 복합 레이블 기준 그룹핑
        grouped = subset_df.groupby("composite_label")

        # 각 복합 레이블 그룹에 대해 train/val/test 분할
        for label, group in grouped:
            indices = group.index.tolist()
            n_samples = len(indices)

            if n_samples < 3:
                print(f"  경고: {label} 샘플 수 {n_samples}개 → 전부 train에 배정")
                df.loc[indices, "split"] = "train"
                continue

            # 분할 개수 계산
            n_test = max(1, int(n_samples * test_ratio))
            n_val = max(1, int(n_samples * val_ratio))
            n_train = n_samples - n_val - n_test

            # train 샘플이 최소 1개 이상 되도록 보정
            if n_train <= 0:
                n_train = 1
                n_val = min(n_val, n_samples - 1)
                n_test = n_samples - n_train - n_val

            # 인덱스 섞어서 분할
            random.shuffle(indices)
            test_idx = indices[:n_test]
            val_idx = indices[n_test:n_test + n_val]
            train_idx = indices[n_test + n_val:]

            # split 레이블 할당
            df.loc[test_idx, "split"] = "test"
            df.loc[val_idx, "split"] = "val"
            df.loc[train_idx, "split"] = "train"

            print(f"  {label}: {n_train} train, {n_val} val, {n_test} test")

    return df


def save_class_files(df):
    """클래스(레이블) 목록을 JSON 파일로 저장하는 함수"""
    # 이동(move) 레이블 목록 (NONE 및 NaN 제외)
    move_classes = sorted([
        c for c in df["move_label"].unique()
        if pd.notna(c) and c != "NONE"
    ])

    # 행동(act) 레이블 목록 (NONE 및 NaN 제외)
    act_classes = sorted([
        c for c in df["act_label"].unique()
        if pd.notna(c) and c != "NONE"
    ])

    # JSON 파일로 저장
    with open(SPLITS_DIR / "class_move.json", "w", encoding="utf-8") as f:
        json.dump(move_classes, f, ensure_ascii=False, indent=2)

    with open(SPLITS_DIR / "class_act.json", "w", encoding="utf-8") as f:
        json.dump(act_classes, f, ensure_ascii=False, indent=2)

    print(f"\n클래스 파일 저장 완료:")
    print(f"  class_move.json: {move_classes}")
    print(f"  class_act.json: {act_classes}")

    return move_classes, act_classes


def analyze_splits(df):
    """train/val/test 분할 결과를 간단히 분석하고 통계를 출력하는 함수"""
    print("\n" + "=" * 60)
    print("데이터 분할 결과 분석")
    print("=" * 60)

    # 전체 분포
    total_counts = df["split"].value_counts()
    print(f"\n전체 split 분포:")
    for split, count in total_counts.items():
        percentage = count / len(df) * 100
        print(f"  {split}: {count}개 샘플 ({percentage:.1f}%)")

    # subset 기준 분포
    print(f"\nsubset 기준 분포:")
    pivot = pd.crosstab(df["subset"], df["split"])
    print(pivot)

    # train 세트 내 이동 레이블 통계
    print(f"\n이동(move) 레이블 분포 (train):")
    train_move = df[df["split"] == "train"]["move_label"].value_counts()
    print(train_move)

    # train 세트 내 행동 레이블 통계
    print(f"\n행동(act) 레이블 분포 (train):")
    train_act = df[df["split"] == "train"]["act_label"].value_counts()
    print(train_act)

    # C_fusion 의 test 분포 확인
    print(f"\nC_fusion 테스트 세트 분포:")
    cfusion_test = df[(df["subset"] == "C_fusion") & (df["split"] == "test")]
    if len(cfusion_test) > 0:
        print(f"  샘플 수: {len(cfusion_test)}")
        print(f"  환경(env) 분포:")
        print(cfusion_test["env"].value_counts())
        print(f"  이동(move) 레이블 분포:")
        print(cfusion_test["move_label"].value_counts())
        print(f"  행동(act) 레이블 분포:")
        print(cfusion_test["act_label"].value_counts())
    else:
        print("  경고: C_fusion 테스트 세트가 비어 있습니다!")


def main():
    """메인 함수: labels.csv를 train/val/test로 분할하고 통계 및 클래스 파일을 생성"""
    parser = argparse.ArgumentParser(description="train/val/test 데이터셋 분할 스크립트")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="검증(Validation) 세트 비율 (기본값: 0.2)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="테스트(Test) 세트 비율 (기본값: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본값: 42)")

    args = parser.parse_args()

    print("=" * 60)
    print("데이터 분할 도구 실행")
    print("=" * 60)
    print(f"설정값: val_ratio={args.val_ratio}, test_ratio={args.test_ratio}, seed={args.seed}")

    # labels.csv 존재 여부 확인
    if not LABELS.exists():
        print(f"오류: labels.csv 파일이 {LABELS} 경로에 존재하지 않습니다.")
        print("먼저 data_preparation.py를 실행하여 labels.csv를 생성해 주세요.")
        return

    # labels.csv 로드
    df = pd.read_csv(LABELS)
    print(f"\nlabels.csv 로드 완료: 총 {len(df)}행")

    # 필요한 컬럼 존재 여부 확인
    required_columns = ["subset", "relpath", "env", "move_label", "act_label"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"오류: 다음 필수 컬럼이 누락되어 있습니다: {missing_cols}")
        return

    # 분층 분할 수행
    print("\n분층(stratified) 데이터 분할을 시작합니다...")
    df_with_splits = make_stratified_splits(
        df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # 분할 결과 저장
    print(f"\n분할 결과를 {SPLITS_DIR} 폴더에 저장합니다...")

    # 전체(full) 파일 저장
    df_with_splits.to_csv(SPLITS_DIR / "full.csv", index=False, encoding="utf-8")

    # train / val / test 각각 저장
    for split_name in ["train", "val", "test"]:
        split_df = df_with_splits[df_with_splits["split"] == split_name]
        split_df.to_csv(SPLITS_DIR / f"{split_name}.csv", index=False, encoding="utf-8")
        print(f"  {split_name}.csv: {len(split_df)}개 샘플")

    # 클래스 파일 저장
    move_classes, act_classes = save_class_files(df_with_splits)

    # 분할 결과 분석 출력
    analyze_splits(df_with_splits)

    print("\n" + "=" * 60)
    print("데이터 분할 완료!")
    print("=" * 60)

    # 각 split 에서 한 개씩 샘플 예시 출력
    print("\n샘플 예시:")
    for split in ["train", "val", "test"]:
        split_df = df_with_splits[df_with_splits["split"] == split]
        if len(split_df) > 0:
            sample = split_df.iloc[0]
            print(f"  {split}: {sample['relpath']} -> {sample['move_label']} + {sample['act_label']}")


if __name__ == "__main__":
    main()
