import sys
import subprocess
import os
import signal
import time
from pathlib import Path

# 현재 파일(run_both.py)이 있는 폴더를 기준으로 경로 설정
BASE = Path(__file__).resolve().parent

# 같이 실행할 스크립트들 (손 제스처 + 음성)
SCRIPTS = [
    BASE / "send_hand_cmd_latefusion.py",
    BASE / "send_voice_cmd_latefusion.py",
]


def launch_same_window():
    """같은 터미널 창에서 두 개의 파이썬 스크립트를 실행하는 함수"""
    procs = []
    for script in SCRIPTS:
        # sys.executable: 지금 파이썬 실행 파일 그대로 사용
        p = subprocess.Popen([sys.executable, str(script)], cwd=str(BASE))
        print(f"[LAUNCHED] {script.name}  PID={p.pid}")  # 실행된 프로세스 정보 출력
        procs.append(p)
    return procs


def main():
    procs = launch_same_window()
    print("\n>>> 시작합니다.\n")

    try:
        # 두 자식 프로세스가 아직 살아있는 동안 계속 체크
        while any(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        # Ctrl+C 입력 들어오면 여기로 옴
        print("\n[STOP] Ctrl+C 감지, 자식 프로세스 종료 중...")
        for p in procs:
            try:
                if p.poll() is None:  # 아직 안 죽었으면
                    if os.name == "nt":
                        # 윈도우
                        p.terminate()
                    else:
                        # 리눅스 / 맥
                        os.kill(p.pid, signal.SIGTERM)
            except:
                # 혹시 에러 나면 그냥 무시 (여기서 에러 때문에 크래시 안 나게)
                pass

        # 깔끔하게 종료될 수 있도록 잠깐 대기
        for p in procs:
            try:
                p.wait(timeout=2)
            except:
                pass
    finally:
        print("[DONE] 프로그램을 종료합니다.")


if __name__ == "__main__":
    # 실행할 스크립트들이 실제로 존재하는지 먼저 확인
    missing = [s for s in SCRIPTS if not s.exists()]
    if missing:
        print("[ERROR] 스크립트를 찾을 수 없습니다:", ", ".join(str(m) for m in missing))
        sys.exit(1)

    main()