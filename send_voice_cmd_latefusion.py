# send_voice_cmd_latefusion_fast.py  (더 빠른 응답 버전)
import socket, speech_recognition as sr, time

UNITY_ADDR = ("127.0.0.1", 5052)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send_v(cmd: str):
    """음성으로 인식된 명령을 Unity 쪽으로 UDP 전송"""
    sock.sendto(f"V:{cmd}".encode("utf-8"), UNITY_ADDR)
    print("[SEND][V]", cmd)


r = sr.Recognizer()
mic = sr.Microphone()

print("음성 인식을 시작합니다. (STOP 없이 빠른 전환 모드)")

# 인식된 텍스트 안에 key 단어가 포함되면 해당 value로 명령을 보냄
CMD_MAP = {
    # 이동
    "forward": "FORWARD", "ahead": "FORWARD", "앞으로": "FORWARD", "向前": "FORWARD", "前进": "FORWARD",
    "back": "BACKWARD", "뒤로": "BACKWARD", "向后": "BACKWARD", "后退": "BACKWARD",
    "left": "LEFT", "왼쪽": "LEFT", "向左": "LEFT", "左转": "LEFT",
    "right": "RIGHT", "오른쪽": "RIGHT", "向右": "RIGHT", "右转": "RIGHT",

    # 액션 (순간적으로 한 번만 발동되는 타입)
    "attack": "ATTACK", "공격": "ATTACK", "攻击": "ATTACK",
    "defend": "DEFEND", "shield": "DEFEND", "방어": "DEFEND", "防御": "DEFEND", "举盾": "DEFEND",
    "jump": "JUMP", "점프": "JUMP", "跳": "JUMP", "跳跃": "JUMP",

    # 추가 별칭
    "skill": "ATTACK", "스킬": "ATTACK",
}

move_state = ""          # 현재 이동 상태 (FORWARD/BACKWARD/LEFT/RIGHT 중 하나)
MOVE_COOLDOWN = 0.15     # 이동 방향을 너무 자주 바꾸지 않도록 하는 최소 간격
ACTION_COOLDOWN = 0.05   # 공격/방어/점프 같은 액션 간 간격
t_last_move = 0.0
t_last_action = 0.0

with mic as source:
    # 시작할 때만 주변 소음을 기준으로 잡고, 이후에는 threshold를 고정해서 쓰기
    r.dynamic_energy_threshold = False   # 매 프레임마다 동적으로 바꾸지 않도록 비활성화
    r.energy_threshold = 250             # 기본 노이즈 기준값 (필요하면 직접 조절 가능)

    # 주변 소음 측정 시간을 짧게 (원래 0.4 정도에서 0.3으로 축소)
    r.adjust_for_ambient_noise(source, duration=0.3)

    # 문장 끝(침묵) 인식 기준을 좀 더 공격적으로 줄이기
    r.pause_threshold = 0.25          # 0.25초 정도 조용하면 한 문장이 끝났다고 봄 (원래 0.6)
    r.non_speaking_duration = 0.2     # 앞뒤로 허용되는 짧은 침묵 (원래 0.4)

# 한 번에 녹음하는 최대 길이 (명령어 하나만 말할 거라서 짧게)
PHRASE_TIME_LIMIT = 1.2  # 필요하면 0.8 ~ 1.0 사이로 더 줄여도 됨

# 실제로 사용하는 언어들 (여기서 필요 없는 언어는 빼면 속도가 조금 더 좋아짐)
langs = ("ko-KR", "zh-CN", "en-US")

while True:
    try:
        print("\n말하세요 (음성 명령)…")
        with mic as source:
            # timeout: 말이 시작되지 않고 기다리는 최대 시간
            audio = r.listen(
                source,
                timeout=2.0,               # 2초 안에 말을 시작하지 않으면 이번 턴 스킵
                phrase_time_limit=PHRASE_TIME_LIMIT
            )

        text = ""
        cmd = None

        # 여러 언어를 순서대로 시도하면서, 먼저 명령어를 잘 찾은 언어를 채택
        for lang in langs:
            try:
                cand = r.recognize_google(audio, language=lang)
                print(f"[ASR] {lang}: {cand}")

                low = cand.lower()
                for k, v in CMD_MAP.items():
                    if k in low:
                        cmd = v
                        text = cand
                        break

                if cmd is not None:
                    # 현재 언어에서 이미 명령을 찾았으면 다른 언어는 더 시도 안 함
                    break

            except Exception as e:
                # 인식 실패 시에는 다음 언어로 넘어감
                print(f"[ASR] {lang} 실패: {e.__class__.__name__}")

        if not cmd:
            print("유효한 명령을 듣지 못했습니다.")
            continue

        now = time.time()

        # ----- 액션 계열: 쿨타임만 지나면 거의 바로바로 발동 -----
        if cmd in ("ATTACK", "DEFEND", "JUMP"):
            if now - t_last_action >= ACTION_COOLDOWN:
                send_v(cmd)
                t_last_action = now
            continue

        # ----- 이동 계열: STOP 없이 방향만 바꿔서 전송 -----
        if cmd in ("FORWARD", "BACKWARD", "LEFT", "RIGHT"):
            # 아직 이동 상태가 없거나, 현재 방향과 다를 때만 전송
            if move_state == "" or cmd != move_state:
                if now - t_last_move >= MOVE_COOLDOWN:
                    send_v(cmd)
                    move_state = cmd
                    t_last_move = now
            else:
                print("같은 방향 명령은 무시합니다.")

    except sr.WaitTimeoutError:
        # 지정한 timeout 동안 말을 시작하지 않으면 여기로 들어옴
        print("시간 초과: 이번 턴에는 아무도 말하지 않았습니다.")
        continue

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        break
