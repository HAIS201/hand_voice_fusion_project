# -*- coding: utf-8 -*-
"""
카메라 + 학습된 Early / Late Fusion GRU 모델 + 마이크 음성
실시간으로 손 제스처(FORWARD / BACKWARD / LEFT / RIGHT / STOP)
와 음성 명령(ATTACK / DEFEND)을 예측하고, UDP 를 통해 Unity 로 전송하는 스크립트.

- 제스처 (GRU) -> G:XXXX
- 음성 (ASR)   -> V:XXXX

주의:
- 실시간 테스트에서는 Mel-spectrogram 을 계산하지 않기 때문에
  오디오 모달리티 입력은 항상 0 으로 두며,
  이는 "손 제스처만 사용하는 GRU" 와 동일한 효과이다.
- 음성 명령은 별도의 ASR(SpeechRecognition) 로 인식하여 보낸다.
"""

import argparse
from collections import deque
from pathlib import Path
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import socket
import speech_recognition as sr

# --------- 경로 & 디바이스 설정 ---------
ROOT = Path(r"D:\pycharm\hand\fusion_project")
OUT_DIR = ROOT / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unity 측 UDPReceiver 가 기다리는 주소
UNITY_ADDR = ("127.0.0.1", 5052)

SEQ_LEN = 60                    # 학습과 동일: 60 프레임 시퀀스
DISPLAY_W, DISPLAY_H = 640, 360 # 카메라 표시 창 크기

# 제스처 + 음성 공용 UDP 소켓
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# ================== 유틸: 시드 고정 & 프레임 정규화 ==================
def seed_all(seed: int = 42):
    """난수 시드를 고정하여 실험 재현성을 높이는 함수"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_frame(pts: np.ndarray) -> np.ndarray:
    """
    오프라인 특성 추출 스크립트와 동일한 정규화 함수.

    Args:
        pts: (21, 3) 형태의 좌표 배열, 픽셀 좌표 (x, y) 와 z 값

    처리:
        - 손목(랜드마크 0)을 원점으로 평행 이동
        - 손바닥 폭(랜드마크 5 와 17 사이 거리)으로 스케일 정규화

    Returns:
        63 차원 벡터 (21 * 3)
    """
    wrist = pts[0, :2]
    palm = np.linalg.norm(pts[5, :2] - pts[17, :2]) + 1e-6
    xy = (pts[:, :2] - wrist) / palm
    z = pts[:, 2:3]
    normed = np.concatenate([xy, z], axis=1)  # (21, 3)
    return normed.reshape(-1).astype(np.float32)  # (63,)


def extract_hand_feature_from_frame(frame, hands_detector):
    """
    MediaPipe Hands 로 한 프레임에서 63 차원 손 제스처 특징을 추출.

    Args:
        frame: BGR 영상 (H, W, 3)
        hands_detector: mediapipe.solutions.hands.Hands 객체

    Returns:
        feat: (63,) float32 벡터
        detected: 손이 검출되었는지 여부(bool)
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]  # 첫 번째 손만 사용
        pts = []
        for p in lm.landmark:
            pts.append([p.x * w, p.y * h, p.z])
        pts = np.array(pts, dtype=np.float32)  # (21, 3)
        feat = normalize_frame(pts)
        return feat, True

    # 손이 검출되지 않은 경우 0 벡터 사용
    return np.zeros((63,), dtype=np.float32), False


# ================== GRU 모델 구조 (학습 코드와 동일) ==================
class GRUEncoder(nn.Module):
    """학습 스크립트에서 사용한 GRUEncoder 와 동일한 구조"""

    def __init__(self, input_dim, hidden=128, num_layers=1, bidir=True, drop=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        self.dropout = nn.Dropout(drop)
        self.out_dim = hidden * (2 if bidir else 1)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.gru(x)        # (B, T, H * dir)
        h = out[:, -1, :]           # 마지막 프레임의 hidden state 사용
        return self.dropout(h)      # (B, out_dim)


class EarlyFusionGRU(nn.Module):
    """early_fusion 학습 스크립트와 동일한 EarlyFusionGRU"""

    def __init__(self, num_move, num_act, drop=0.1):
        super().__init__()
        self.enc_g = GRUEncoder(63, hidden=128, bidir=True, drop=drop)
        self.enc_a = GRUEncoder(64, hidden=128, bidir=True, drop=drop)
        fusion_in = self.enc_g.out_dim + self.enc_a.out_dim  # 256 + 256 = 512
        self.fuse = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.head_move = nn.Linear(128, num_move)
        self.head_act = nn.Linear(128, num_act)

        self._init_weights()

    def _init_weights(self):
        """Xavier 초기화로 가중치 설정"""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, g_seq, a_seq):
        hg = self.enc_g(g_seq)
        ha = self.enc_a(a_seq)
        h = self.fuse(torch.cat([hg, ha], dim=-1))
        lm = self.head_move(h)
        la = self.head_act(h)
        return lm, la


class LateFusionGRU(nn.Module):
    """
    late_fusion 학습 스크립트와 동일한 LateFusionGRU.

    - 제스처/오디오를 각각 인코딩한 뒤,
      fusion=True 인 경우 두 특징을 concat 하여 공동 분류.
    - fusion=False 인 경우, 제스처/오디오 각각의 단독 head 사용.
    """

    def __init__(self, num_move, num_act, drop=0.1):
        super().__init__()
        self.enc_g = GRUEncoder(63, hidden=128, bidir=True, drop=drop)
        self.enc_a = GRUEncoder(64, hidden=128, bidir=True, drop=drop)

        # 제스처 단일 모달 head (이동만 예측)
        self.head_g_move = nn.Sequential(
            nn.Linear(self.enc_g.out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_move),
        )

        # 오디오 단일 모달 head (행동만 예측)
        self.head_a_act = nn.Sequential(
            nn.Linear(self.enc_a.out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_act),
        )

        # 융합 head (학습 시 주로 사용하는 부분)
        fusion_dim = self.enc_g.out_dim + self.enc_a.out_dim
        self.fusion_head_move = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_move),
        )
        self.fusion_head_act = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(128, num_act),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 초기화로 가중치 설정"""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, g_seq, a_seq, fusion=True):
        hg = self.enc_g(g_seq)
        ha = self.enc_a(a_seq)
        if fusion:
            # 후기 융합: 두 모달 특징을 concat 후 분류
            h_f = torch.cat([hg, ha], dim=-1)
            lm = self.fusion_head_move(h_f)
            la = self.fusion_head_act(h_f)
            return lm, la
        else:
            # 단일 모달 헤드 사용
            lm_g = self.head_g_move(hg)
            la_a = self.head_a_act(ha)
            return lm_g, la_a


# ================== 체크포인트 로드 ==================
def load_early_model():
    """학습된 EarlyFusion 모델과 클래스 리스트 로드"""
    ckpt_path = OUT_DIR / "early_gru_best.pth"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    moves = ckpt["moves"]
    acts = ckpt["acts"]
    args_dict = ckpt.get("args", {})
    drop = args_dict.get("drop", 0.3)

    model = EarlyFusionGRU(num_move=len(moves), num_act=len(acts), drop=drop).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[OK] EarlyFusion 모델 로드: {ckpt_path}")
    print("  move 클래스:", moves)
    print("  act  클래스:", acts)
    return model, moves, acts


def load_late_model():
    """학습된 LateFusion 모델과 클래스 리스트 로드"""
    ckpt_path = OUT_DIR / "late_gru_best.pth"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    moves = ckpt["moves"]
    acts = ckpt["acts"]
    args_dict = ckpt.get("args", {})
    drop = args_dict.get("drop", 0.3)

    model = LateFusionGRU(num_move=len(moves), num_act=len(acts), drop=drop).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[OK] LateFusion 모델 로드: {ckpt_path}")
    print("  move 클래스:", moves)
    print("  act  클래스:", acts)
    return model, moves, acts


# ================== UDP 전송 ==================
def send_g_cmd(move_cmd, act_cmd):
    """
    GRU/제스처 쪽에서 Unity 로 보내는 패킷.

    - move_cmd: FORWARD/BACKWARD/LEFT/RIGHT/STOP/NONE
    - act_cmd : ATTACK/DEFEND/NONE

    둘 다 "G:" 프리픽스를 사용한다.
    """
    if move_cmd and move_cmd != "NONE":
        sock.sendto(f"G:{move_cmd}".encode("utf-8"), UNITY_ADDR)
        print("[SEND][G-MOVE]", move_cmd)

    if act_cmd and act_cmd != "NONE":
        sock.sendto(f"G:{act_cmd}".encode("utf-8"), UNITY_ADDR)
        print("[SEND][G-ACT ]", act_cmd)


def send_v_cmd(cmd: str):
    """
    ASR(음성 인식) 결과를 Unity 로 전송하는 함수.

    - ATTACK / DEFEND / FORWARD / BACKWARD / LEFT / RIGHT / STOP ...
    - "V:" 프리픽스를 사용한다.
    """
    if not cmd:
        return
    sock.sendto(f"V:{cmd}".encode("utf-8"), UNITY_ADDR)
    print("[SEND][V]", cmd)


# ================== 음성 인식 (백그라운드 스레드) ==================
# 필요하면 중/영/한 키워드를 더 추가할 수 있다.
VOICE_CMD_MAP = {
    # 공격
    "attack": "ATTACK",
    "공격": "ATTACK",
    "어택": "ATTACK",
    "攻击": "ATTACK",  # 중국어 키워드 그대로 두지만, 설명은 한글

    # 방어
    "defend": "DEFEND",
    "디펜드": "DEFEND",
    "방어": "DEFEND",
    "防御": "DEFEND",
}


def start_voice_thread(asr_lang: str = "ko-KR"):
    """
    SpeechRecognition 으로 마이크를 백그라운드에서 계속 듣다가,
    특정 키워드를 인식하면 V:XXXX 명령을 전송하는 스레드를 시작한다.

    Args:
        asr_lang: Google Speech API 에 전달할 언어 코드
                  예) "ko-KR", "zh-CN", "en-US"
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # 주변 소음 기준으로 마이크 보정
    with mic as source:
        print("[VOICE] 마이크 보정 중... 잠시만 조용히 해 주세요.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[VOICE] 보정 완료, 백그라운드 청취 시작 (종료: Ctrl+C 또는 창 종료)")

    def callback(recognizer_, audio):
        """음성이 들어올 때마다 호출되는 콜백 함수"""
        try:
            # 기본값은 한국어, 필요하면 main 함수 인자로 변경
            text = recognizer_.recognize_google(audio, language=asr_lang)
            text = text.strip()
            print(f"[VOICE] 인식 결과: {text}")
        except Exception as e:
            print("[VOICE] 인식 실패:", e)
            return

        lower = text.lower()

        # 단순 키워드 매칭으로 명령 결정
        for key, cmd in VOICE_CMD_MAP.items():
            if key in lower or key in text:
                send_v_cmd(cmd)
                break

    # listen_in_background 는 별도의 스레드에서 마이크를 청취한다.
    stop_func = recognizer.listen_in_background(mic, callback)
    return stop_func


# ================== 메인 루프 (카메라 + GRU) ==================
def cam_gru_loop(args):
    """카메라 프레임을 읽어 GRU 로 예측하고, Unity 로 명령을 보내는 메인 루프"""
    seed_all(42)

    # 모드 선택에 따라 모델 로드 방식 결정
    if args.mode == "early":
        model, moves, acts = load_early_model()
        mode_str = "Early Fusion (gesture + zero-audio)"
        use_fusion = True
    elif args.mode == "late":
        model, moves, acts = load_late_model()
        mode_str = "Late Fusion (fusion head, gesture + zero-audio)"
        use_fusion = True
    else:
        # late_g_only: LateFusion 의 제스처 단일 head 만 사용하여 이동만 예측
        model, moves, acts = load_late_model()
        mode_str = "Late Fusion (gesture-only move head, fusion=False)"
        use_fusion = False

    print(f"\n[MODE] {mode_str}")
    print(f"[INFO] Unity UDP 타겟 주소: {UNITY_ADDR}\n")

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # MediaPipe Hands 초기화
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 최근 SEQ_LEN 프레임을 저장하는 버퍼
    g_buffer = deque(maxlen=SEQ_LEN)

    # 마지막으로 전송한 명령 (동일 명령 반복 전송을 막기 위함)
    last_move_cmd = "NONE"
    last_act_cmd = "NONE"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 한 프레임에서 63 차원 손 특징 추출
            feat63, detected = extract_hand_feature_from_frame(frame, hands_detector)
            g_buffer.append(feat63)

            move_label = "NONE"
            act_label = "NONE"
            move_conf = 0.0
            act_conf = 0.0

            # 버퍼가 60 프레임에 도달했을 때만 GRU 에 넣어서 예측
            if len(g_buffer) == SEQ_LEN:
                g_seq = np.stack(list(g_buffer), axis=0)  # (60, 63)
                g_seq_t = torch.from_numpy(g_seq).unsqueeze(0).to(DEVICE).float()  # (1, 60, 63)

                # 오디오 모달리티는 현재 0 으로 채운 시퀀스를 사용
                a_seq = np.zeros((SEQ_LEN, 64), dtype=np.float32)
                a_seq_t = torch.from_numpy(a_seq).unsqueeze(0).to(DEVICE).float()  # (1, 60, 64)

                with torch.no_grad():
                    if isinstance(model, EarlyFusionGRU):
                        lm, la = model(g_seq_t, a_seq_t)
                    else:
                        lm, la = model(g_seq_t, a_seq_t, fusion=use_fusion)

                    pm = F.softmax(lm, dim=-1)
                    pa = F.softmax(la, dim=-1)

                mi = int(pm.argmax(dim=-1).item())
                move_label = moves[mi]
                move_conf = float(pm[0, mi].item())

                if len(acts) > 0:
                    ai = int(pa.argmax(dim=-1).item())
                    act_label = acts[ai]
                    act_conf = float(pa[0, ai].item())
                else:
                    act_label = "NONE"
                    act_conf = 0.0

                # ---- Unity 로 명령 전송 (softmax 임계값 사용) ----
                if move_label != "NONE" and move_conf >= args.move_thr:
                    if move_label != last_move_cmd:
                        send_g_cmd(move_label, "NONE")
                        last_move_cmd = move_label

                # early/late 모드에서만 GRU 로 행동 명령 전송
                if args.mode in ("early", "late") and act_label != "NONE" and act_conf >= args.act_thr:
                    if act_label != last_act_cmd:
                        send_g_cmd("NONE", act_label)
                        last_act_cmd = act_label

            # ---- 화면 HUD 표시 ----
            hud = frame.copy()
            cv2.putText(
                hud,
                f"mode={args.mode}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                hud,
                f"len={len(g_buffer)}/{SEQ_LEN}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                hud,
                f"Move: {move_label} ({move_conf:.2f})",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if args.mode in ("early", "late"):
                cv2.putText(
                    hud,
                    f"Act : {act_label} ({act_conf:.2f})",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )

            disp = cv2.resize(hud, (DISPLAY_W, DISPLAY_H))
            cv2.imshow("Cam GRU + Voice -> Unity", disp)

            # 'esc' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        hands_detector.close()
        cv2.destroyAllWindows()
        print("[DONE] 카메라/GRU 루프 종료")


# ================== main ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="카메라 + GRU + 음성 인식 Unity 데모")
    parser.add_argument(
        "--mode",
        type=str,
        default="early",
        choices=["early", "late", "late_g_only"],
        help=(
            "early:  early_gru_best.pth 사용 (조기 융합)\n"
            "late:   late_gru_best.pth 사용 (후기 융합, fusion head)\n"
            "late_g_only: LateFusion 의 제스처 전용 head 만 사용하여 이동만 예측"
        ),
    )
    parser.add_argument(
        "--move_thr",
        type=float,
        default=0.6,
        help="이동 명령을 전송하기 위한 최소 softmax confidence 값",
    )
    parser.add_argument(
        "--act_thr",
        type=float,
        default=0.6,
        help="행동 명령을 전송하기 위한 최소 softmax confidence 값 (early/late 모드에서 사용)",
    )
    parser.add_argument(
        "--asr_lang",
        type=str,
        default="ko-KR",
        help="음성 인식에 사용할 언어 코드 (예: 'ko-KR', 'zh-CN', 'en-US')",
    )
    args = parser.parse_args()

    # 백그라운드 음성 인식 스레드 시작
    stop_voice = start_voice_thread(asr_lang=args.asr_lang)

    try:
        # 메인 스레드: 카메라 + GRU 루프 실행
        cam_gru_loop(args)
    finally:
        # 음성 인식 중지 및 소켓 종료
        if stop_voice is not None:
            stop_voice(wait_for_stop=False)
        sock.close()
        print("[DONE] 프로그램 종료")
