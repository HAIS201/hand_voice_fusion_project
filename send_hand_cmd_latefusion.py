# send_hand_cmd_latefusion_center.py
# 오른손: 화면 가운데 쪽의 기준점으로 이동 / 왼손: 공격·방어·점프 액션
import cv2, socket, time, math
from cvzone.HandTrackingModule import HandDetector

# ---------------- 기본 설정 ----------------
WIDTH, HEIGHT = 1280, 720
DX_RATIO, DY_RATIO = 0.18, 0.16         # 기준점에서 얼마나 벗어나야 방향으로 인식할지 (비율, 작을수록 더 민감)

# 원래는 오른쪽 반쪽만 이동 영역으로 썼는데,
# 이제는 화면 전체에서 오른손을 쓰되, 기준점만 살짝 오른쪽(0.6W)으로 둠
CTR_X = int(WIDTH * 0.60)               # 오른손 이동 기준 X 좌표
CTR_Y = int(HEIGHT * 0.50)              # 이동 기준 Y 좌표 (화면 세로 중앙)

# 비율을 실제 픽셀로 변환
TX = int(DX_RATIO * WIDTH)
TY = int(DY_RATIO * HEIGHT)

DISPLAY_W = 640
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
    """손가락이 펴져 있는지 간단히 체크 (손목과의 거리 비교)"""
    return dist(lm[tip], lm[0]) > dist(lm[pip], lm[0])


def classify_hand(hand):
    """
    반환:
      Left 손  -> 'ATTACK' / 'DEFEND' / 'JUMP'
      Right 손 -> 'FORWARD' / 'BACKWARD' / 'LEFT' / 'RIGHT'
      그 외    -> None
    """
    lm = hand["lmList"]
    htype = hand["type"]
    if not lm:
        return None

    # MediaPipe 랜드마크 인덱스
    TH_TIP, TH_MCP = 4, 2
    IN_TIP, IN_PIP = 8, 6
    MI_TIP, MI_PIP = 12, 10
    RI_TIP, RI_PIP = 16, 14
    LI_TIP, LI_PIP = 20, 18

    th  = is_extended(lm, TH_TIP, TH_MCP)
    ind = is_extended(lm, IN_TIP, IN_PIP)
    mid = is_extended(lm, MI_TIP, MI_PIP)
    rin = is_extended(lm, RI_TIP, RI_PIP)
    lit = is_extended(lm, LI_TIP, LI_PIP)
    ext_count = sum([th, ind, mid, rin, lit])

    # 엄지-검지 거리가 가까우면 pinch로 판단
    pinch = dist(lm[4], lm[8]) / (dist(lm[5], lm[17]) + 1e-6) < 0.35

    # ---------- 왼손: 액션 ----------
    if htype == "Left":
        # 손가락 거의 다 접으면 공격
        if ext_count <= 1:
            return "ATTACK"
        # 손가락 대부분 펴면 방어
        if ext_count >= 4:
            return "DEFEND"
        # pinch 제스처는 점프
        if pinch:
            return "JUMP"
        return None

    # ---------- 오른손: 이동 ----------
    if htype == "Right":
        # 오른손이면 화면 어디에 있든, 기준점(CTR_X, CTR_Y) 기준으로만 판단
        cx, cy = lm[9][0], lm[9][1]   # 중지 MCP 근처를 손의 대표 위치로 사용

        # 기준점에서 얼마나 벗어났는지
        dx, dy = cx - CTR_X, cy - CTR_Y

        # 수평/수직 중 더 크게 벗어난 방향으로 커맨드 결정
        if abs(dx) > abs(dy):
            if dx > TX:
                return "RIGHT"
            if dx < -TX:
                return "LEFT"
        else:
            # 위쪽으로 가면 앞으로, 아래로 가면 뒤로
            if dy < -TY:
                return "FORWARD"
            if dy > TY:
                return "BACKWARD"
        return None

    return None


# ---------------- 메인 루프 ----------------
move_state   = ""     # 현재 이동 상태 (FORWARD/BACKWARD/LEFT/RIGHT)
last_action  = ""     # 마지막 왼손 액션 (ATTACK/DEFEND/JUMP)
ACTION_COOLDOWN = 0.2 # 왼손 액션 쿨타임 (초 단위)
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

    # ---------- 오른손 이동: 상태가 바뀔 때만 전송 (STOP은 따로 안 보냄) ----------
    if right_cmd in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
        if right_cmd != move_state:
            sock.sendto(f"G:{right_cmd}".encode(), UNITY_ADDR)
            print("Send(G move):", right_cmd)
            move_state = right_cmd

    # ---------- 왼손 액션: 바뀔 때 + 쿨타임 이후에 한 번만 전송 ----------
    now = time.time()
    if left_cmd in ("ATTACK", "DEFEND", "JUMP"):
        if (left_cmd != last_action) and (now - t_last_action > ACTION_COOLDOWN):
            sock.sendto(f"G:{left_cmd}".encode(), UNITY_ADDR)
            print("Send(G action):", left_cmd)
            last_action = left_cmd
            t_last_action = now
    elif left_cmd is None:
        # 왼손이 인식 안 되면 액션 상태 초기화
        last_action = ""

    # ---------------- 화면 HUD ----------------
    # 좌우 구분선 없이, 기준점 + 박스만 보여줌
    cv2.drawMarker(
        img,
        (CTR_X, CTR_Y),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2
    )
    cv2.rectangle(
        img,
        (CTR_X - TX, CTR_Y - TY),
        (CTR_X + TX, CTR_Y + TY),
        (0, 255, 255),
        1
    )

    # 텍스트 HUD
    cv2.putText(
        img,
        f"MOVE:{move_state or '-'}  ACT:{last_action or '-'}",
        (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    cv2.putText(
        img,
        "이동",
        (CTR_X - 80, max(CTR_Y - 30, 20)),   # 기준점 근처에 라벨 표시
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Gesture->UDP (Right hand move center near middle, No STOP)",
        cv2.resize(img, (DISPLAY_W, DISPLAY_H))
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
