import csv
import re
from pathlib import Path

# 프로젝트 루트 폴더
ROOT = Path(r"D:\pycharm\hand\fusion_project")
DATA = ROOT / "data"
OUT_CSV = ROOT / "labels.csv"

# 환경 코드 매핑
E_MAP = {
    "E1": "BrightQuiet",
    "E2": "BrightFan",
    "E3": "DimQuiet",
    "E4": "DimFan",
}

# C_fusion 템플릿: 코드 -> (손 제스처 이동, 음성 액션)
FUSION_TABLE = {
    "F1": ("FORWARD",  "ATTACK"),
    "F2": ("LEFT",     "ATTACK"),
    "F3": ("RIGHT",    "ATTACK"),
    "F4": ("FORWARD",  "DEFEND"),
    "F5": ("BACKWARD", "DEFEND"),
}

rows = []

# ---------- A_hand ----------
for e in ["E1", "E3"]:
    d = DATA / "A_hand" / e
    if not d.exists():
        continue

    for f in sorted(d.glob("*.mp4")):
        # 예시: FORWARD_01_E1.mp4 / ATTACK_03_E3.mp4
        m = re.match(r"([A-Z]+)_[0-9]+_(E[0-9])\.mp4$", f.name)
        if not m:
            print("[A_hand][WARN] 파일 이름 패턴 불일치, 스킵:", f.name)
            continue

        label = m.group(1)
        env = E_MAP[m.group(2)]

        # 학습 라벨은 6개로 통일: FORWARD/BACKWARD/LEFT/RIGHT/ATTACK/DEFEND
        if label not in {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "ATTACK", "DEFEND"}:
            print("[A_hand][WARN] 지원하지 않는 라벨:", label)
            continue

        rows.append(["A_hand", str(f.relative_to(DATA)), label, env])

print(f"[A_hand] 샘플 수: {sum(1 for r in rows if r[0] == 'A_hand')}")

# ---------- B_voice ----------
for e in ["E1", "E2"]:
    d = DATA / "B_voice" / e
    if not d.exists():
        continue

    for f in sorted(d.glob("*.mp4")):
        # 예시: ATTACK_01_E1.mp4 / DEFEND_03_E2.mp4
        m = re.match(r"(ATTACK|DEFEND)_[0-9]+_(E[0-9])\.mp4$", f.name)
        if not m:
            print("[B_voice][WARN] 파일 이름 패턴 불일치, 스킵:", f.name)
            continue

        label = m.group(1)
        env = E_MAP[m.group(2)]
        rows.append(["B_voice", str(f.relative_to(DATA)), label, env])

print(f"[B_voice] 샘플 수: {sum(1 for r in rows if r[0] == 'B_voice')}")

# ---------- C_fusion ----------
for e in ["E1", "E2", "E3", "E4"]:
    d = DATA / "C_fusion" / e
    if not d.exists():
        continue

    for f in sorted(d.glob("*.mp4")):
        # 예시: F1_01_E1.mp4 이런 형태라고 가정
        m = re.match(r"(F[1-5])_[0-9]+_(E[0-9])\.mp4$", f.name)
        if not m:
            print("[C_fusion][WARN] 파일 이름 패턴 불일치, 스킵:", f.name)
            continue

        fcode = m.group(1)
        env = E_MAP[m.group(2)]

        if fcode not in FUSION_TABLE:
            print("[C_fusion][WARN] FUSION_TABLE에 없는 코드:", fcode)
            continue

        move, act = FUSION_TABLE[fcode]
        # 라벨은 "MOVE+ACTION" 형태로 합쳐서 저장
        rows.append(["C_fusion", str(f.relative_to(DATA)), f"{move}+{act}", env])

print(f"[C_fusion] 샘플 수: {sum(1 for r in rows if r[0] == 'C_fusion')}")

# ---------- CSV로 저장 ----------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as w:
    cw = csv.writer(w)
    cw.writerow(["subset", "relpath", "label", "env"])
    cw.writerows(rows)

print(f"[OK] CSV 저장 완료: {OUT_CSV}  / 총 {len(rows)}행")