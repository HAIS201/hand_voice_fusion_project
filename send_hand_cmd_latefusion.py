# send_hand_cmd_digits.py
# 오른손: 손가락 개수(1~5)로 이동 / 왼손: 공격·방어 (JUMP 제거 버전)

import cv2, socket, time, math
from cvzone.HandTrackingModule import HandDetector

# ---------------- 기본 설정 ----------------
WIDTH, HEIGHT = 1280, 720

# 화면 표시용 크기 (조금 더 작게)
DISPLAY_W = 480
DISPLAY_H = int(HEIGHT * DISPLAY_W / WIDTH)

# ---------------- 카메라 & 네트워크 ----------------
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

detector = HandDetector(maxHands=2, detectionCon=0.8)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UNITY_ADDR = ("127.0.0.1", 5052)

# ---------------- 유틸 함수 ----------------
def dist(a, b):
    """두 점 사이의 유클리드 거리"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_extended(lm, tip, pip):
    """손가락이 펴져 있는지 간단 체크 (손목과의 거리 비교)"""
    return dist(lm[tip], lm[0]) > dist(lm[pip], lm[0])


def classify_hand(hand):
    """
    반환:
      Left 손  -> 'ATTACK' / 'DEFEND'
      Right 손 -> 'FORWARD' / 'BACKWARD' / 'LEFT' / 'RIGHT' / 'STOP'
      그 외    -> None
    """
    lm = hand["lmList"]
    htype = hand["type"]
    if not lm:
        return None

    # MediaPipe 랜드마크 인덱스
    TH_TIP, TH_MCP = 4, 2      # 엄지
    IN_TIP, IN_PIP = 8, 6      # 검지
    MI_TIP, MI_PIP = 12, 10    # 중지
    RI_TIP, RI_PIP = 16, 14    # 약지
    LI_TIP, LI_PIP = 20, 18    # 새끼

    th  = is_extended(lm, TH_TIP, TH_MCP)
    ind = is_extended(lm, IN_TIP, IN_PIP)
    mid = is_extended(lm, MI_TIP, MI_PIP)
    rin = is_extended(lm, RI_TIP, RI_PIP)
    lit = is_extended(lm, LI_TIP, LI_PIP)

    # 엄지를 제외한 4개 손가락 개수만 사용
    ext_4 = sum([ind, mid, rin, lit])

    # ---------- 왼손: 액션 (JUMP 제거) ----------
    if htype == "Left":
        # 손가락 대부분 접힘 → 공격
        if ext_4 <= 1:
            return "ATTACK"
        # 손가락 대부분 펴짐 → 방어
        if ext_4 >= 3:
            return "DEFEND"
        return None

    # ---------- 오른손: 손가락 개수로 이동 ----------
    if htype == "Right":
        ext_4 = sum([ind, mid, rin, lit])

        if th and ext_4 == 4:
            return "STOP"

        if ext_4 == 4:
            return "RIGHT"      # 4손가락 = 오른쪽
        elif ext_4 == 3:
            return "LEFT"       # 3손가락 = 왼쪽
        elif ext_4 == 2:
            return "BACKWARD"   # 2손가락 = 뒤로
        elif ext_4 == 1:
            return "FORWARD"    # 1손가락 = 앞으로
        else:
            # 0개(주먹 등) → 이동 명령 없음
            return None

    return None


# ---------------- 메인 루프 ----------------
move_state   = ""     # 현재 이동 상태 (FORWARD/BACKWARD/LEFT/RIGHT/STOP)
last_action  = ""     # 마지막 왼손 액션 (ATTACK/DEFEND)
ACTION_COOLDOWN = 0.2 # 왼손 액션 쿨타임 (초)
t_last_action  = 0.0

while True:
    ok, img = cap.read()
    if not ok:
        break

    hands, img = detector.findHands(img, flipType=False)

    right_cmd, left_cmd = None, None
    for h in hands:
        c = classify_hand(h)
        if h["type"] == "Right":
            right_cmd = c
        elif h["type"] == "Left":
            left_cmd = c

    # ---------- 오른손 이동: 상태가 바뀔 때만 전송 ----------
    if right_cmd in ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"):
        if right_cmd != move_state:
            sock.sendto(f"G:{right_cmd}".encode(), UNITY_ADDR)
            print("Send(G move):", right_cmd)
            move_state = right_cmd

    # ---------- 왼손 액션: 바뀔 때 + 쿨타임 이후 전송 ----------
    now = time.time()
    if left_cmd in ("ATTACK", "DEFEND"):
        if (left_cmd != last_action) and (now - t_last_action > ACTION_COOLDOWN):
            sock.sendto(f"G:{left_cmd}".encode(), UNITY_ADDR)
            print("Send(G action):", left_cmd)
            last_action = left_cmd
            t_last_action = now
    elif left_cmd is None:
        last_action = ""

    # ---------------- 화면 HUD ----------------
    cv2.putText(
        img,
        f"MOVE:{move_state or '-'}  ACT:{last_action or '-'}",
        (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Gesture->UDP (Right hand 1~5, Left ATTACK/DEFEND)",
        cv2.resize(img, (DISPLAY_W, DISPLAY_H))
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
