# -*- coding: utf-8 -*-
import re
import csv
from pathlib import Path

ROOT = Path(r"D:\pycharm\hand\fusion_project")
DATA = ROOT / "data"
OUT_CSV = ROOT / "labels.csv"

# 환경 코드 (선택 사항, 기록용)
E_MAP = {
    "E1": "BrightQuiet",
    "E2": "BrightFan",
    "E3": "DimQuiet",
    "E4": "DimFan",
}

MOVE_SET = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}
ACT_SET  = {"ATTACK", "DEFEND"}

rows = []

# ---------- A_hand: 손 제스처(이동 전용) ----------
a_dir = DATA / "A_hand"
if a_dir.exists():
    for f in sorted(a_dir.glob("*.mp4")):
        m = re.match(r"(FORWARD|BACKWARD|LEFT|RIGHT|STOP|ATTACK|DEFEND)_[0-9]+_(E[1-4])\.mp4$", f.name)
        if not m:
            print("[A_hand][WARN] 파일명이 규칙과 맞지 않아 건너뜀:", f.name)
            continue
        label = m.group(1)
        env_code = m.group(2)
        env = E_MAP.get(env_code, env_code)

        # 이동 클래스만 사용하고 ATTACK/DEFEND 제스처는 일단 제외
        if label not in MOVE_SET:
            print("[A_hand][SKIP] 이동 클래스가 아니라서 사용하지 않음:", f.name)
            continue

        rows.append([
            "A_hand",
            f"A_hand/{f.name}",
            env,
            label,   # 이동 레이블
            "NONE",  # 행동 레이블
        ])

print("[A_hand] 샘플 개수:", sum(1 for r in rows if r[0] == "A_hand"))

# ---------- B_voice: 음성(행동 전용) ----------
b_dir = DATA / "B_voice"
if b_dir.exists():
    for f in sorted(b_dir.glob("*.mp4")):
        m = re.match(r"(ATTACK|DEFEND)_[0-9]+_(E[1-4])\.mp4$", f.name)
        if not m:
            print("[B_voice][WARN] 파일명이 규칙과 맞지 않아 건너뜀:", f.name)
            continue
        act = m.group(1)
        env_code = m.group(2)
        env = E_MAP.get(env_code, env_code)

        rows.append([
            "B_voice",
            f"B_voice/{f.name}",
            env,
            "NONE",  # 이동 레이블
            act,     # 행동 레이블
        ])

print("[B_voice] 샘플 개수:", sum(1 for r in rows if r[0] == "B_voice"))

# ---------- C_fusion: 이동 + 행동 동시 기록 ----------
c_dir = DATA / "C_fusion"
if c_dir.exists():
    for f in sorted(c_dir.glob("*.mp4")):
        # 예: FORWARD_ATTACK_01_E1.mp4
        m = re.match(r"(FORWARD|BACKWARD|LEFT|RIGHT|STOP)_(ATTACK|DEFEND)_[0-9]+_(E[1-4])\.mp4$", f.name)
        if not m:
            print("[C_fusion][WARN] 파일명이 규칙과 맞지 않아 건너뜀:", f.name)
            continue
        move = m.group(1)
        act  = m.group(2)
        env_code = m.group(3)
        env = E_MAP.get(env_code, env_code)

        rows.append([
            "C_fusion",
            f"C_fusion/{f.name}",
            env,
            move,
            act,
        ])

print("[C_fusion] 샘플 개수:", sum(1 for r in rows if r[0] == "C_fusion"))

# ---------- CSV 파일로 저장 ----------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as w:
    cw = csv.writer(w)
    cw.writerow(["subset", "relpath", "env", "move_label", "act_label"])
    cw.writerows(rows)

print(f"[OK] CSV 저장 완료: {OUT_CSV} (총 {len(rows)}행)")
