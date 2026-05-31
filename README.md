# 손 제스처와 음성 인식을 활용한 다중 모달 융합 연구

## 📌 프로젝트 소개

본 프로젝트는 손 제스처와 음성 명령을 함께 활용하는 다중 모달 실시간 제어 시스템을 설계하고 구현한 연구이다. 게임, VR, 교육용 시뮬레이션과 같은 환경에서는 키보드와 마우스를 벗어나 보다 자연스럽고 직관적인 인터페이스가 요구된다. 이에 따라 본 연구에서는 손 제스처와 음성이라는 서로 다른 입력 모달리티를 결합하여 캐릭터를 제어하는 방식을 구현하였다.

본 시스템에서 손 제스처는 캐릭터의 이동 명령을 담당하고, 음성은 공격 및 방어와 같은 행동 명령을 담당한다. 손 제스처는 방향과 위치 같은 공간적 정보를 표현하는 데 적합하며, 음성은 “공격”, “방어”와 같은 고수준 명령을 빠르게 전달하는 데 적합하다. 따라서 두 모달리티를 함께 사용하면 단일 입력 방식보다 더 자연스럽고 몰입감 있는 상호작용을 제공할 수 있다.

본 프로젝트의 주요 목표는 다음과 같다.

* 웹캠과 마이크를 이용하여 손 제스처 및 음성 데이터를 직접 수집
* MediaPipe Hands와 Mel-spectrogram 기반 특징 추출 파이프라인 구현
* GRU 기반 Early Fusion 및 Late Fusion 모델 학습
* 동일한 데이터셋에서 두 융합 방식의 성능 비교
* Python 인식 시스템과 Unity를 UDP로 연동한 실시간 캐릭터 제어 데모 구현

---

## 🔎 전체 동작 흐름

### 1. 데이터 수집 및 라벨링

직접 촬영한 데이터를 다음 세 가지 subset으로 구성하였다.

* `A_hand`: 손 제스처 중심 데이터
* `B_voice`: 음성 명령 중심 데이터
* `C_fusion`: 손 제스처와 음성이 동시에 포함된 복합 데이터

데이터는 다음과 같은 환경 조건으로 나누어 수집하였다.

* `BrightQuiet`: 밝은 조명 + 조용한 환경
* `BrightFan`: 밝은 조명 + 선풍기 소음 환경
* `DimQuiet`: 어두운 조명 + 조용한 환경
* `DimFan`: 어두운 조명 + 선풍기 소음 환경

전체 메타데이터는 `labels.csv`로 관리하였고, 학습·검증·테스트 분할은 `splits/train.csv`, `splits/val.csv`, `splits/test.csv`로 구성하였다.

---

### 2. 특징 추출

#### 손 제스처 특징

손 제스처 특징 추출에는 MediaPipe Hands를 사용하였다.

* 한 프레임에서 21개 손 랜드마크 추출
* 각 랜드마크는 x, y, z 좌표로 구성
* 손목 landmark를 기준으로 상대 좌표 변환
* 손바닥 너비를 기준으로 스케일 정규화
* 최종 입력 형태: `(60, 63)`

#### 음성 특징

음성 특징 추출에는 Mel-spectrogram을 사용하였다.

* mp4 영상에서 오디오 추출
* mono, 16kHz로 변환
* 무음 구간 처리 및 정규화
* 64개 Mel-bin 사용
* 최종 입력 형태: `(60, 64)`

---

## 🧠 모델 구조

본 연구에서는 GRU 기반 Early Fusion 모델과 Late Fusion 모델을 구현하여 비교하였다.

### Early Fusion GRU

Early Fusion 모델은 제스처 인코더와 음성 인코더의 출력을 결합한 뒤, 하나의 통합 표현을 기반으로 이동 명령과 행동 명령을 예측한다.

* Gesture Encoder: Bi-GRU
* Audio Encoder: Bi-GRU
* Fusion Feature: gesture feature + audio feature
* Output:

  * Move command
  * Act command

Early Fusion은 두 모달리티의 정보를 이른 단계에서 결합하기 때문에 모달 간 상호작용을 직접 반영할 수 있다는 장점이 있다.

### Late Fusion GRU

Late Fusion 모델은 제스처와 음성을 독립적으로 처리한 뒤, 각각의 결과를 기반으로 명령을 예측한다.

* Gesture Encoder → Move Head
* Audio Encoder → Act Head

Late Fusion은 모달별 독립성을 유지하기 때문에 특정 모달이 약해지는 환경에서도 상대적으로 안정적인 처리가 가능하다.

---

## 📊 실험 설정 및 최종 결과

동일한 데이터셋과 동일한 train/validation/test split을 사용하여 Early Fusion과 Late Fusion을 비교하였다.

평가 지표는 다음과 같다.

* `Move Accuracy`: 이동 명령 예측 정확도
* `Act Accuracy`: 행동 명령 예측 정확도
* `C_fusion CSR`: 복합 명령에서 이동과 행동을 모두 맞춘 비율

### 최종 실험 결과

| Model        | Move Accuracy | Act Accuracy | C_fusion CSR |
| ------------ | ------------: | -----------: | -----------: |
| Early Fusion |        0.9818 |       0.8295 |       0.8250 |
| Late Fusion  |        0.9818 |       0.8409 |       0.8250 |

최종 결과에서 두 모델은 동일한 이동 정확도와 C_fusion CSR을 보였다. 행동 정확도에서는 Late Fusion이 Early Fusion보다 소폭 높은 성능을 보였다.

### 환경별 C_fusion CSR

| Environment | Early Fusion | Late Fusion |
| ----------- | -----------: | ----------: |
| BrightFan   |       0.8500 |      0.8500 |
| BrightQuiet |       0.9500 |      0.9500 |
| DimFan      |       0.6364 |      0.6364 |
| DimQuiet    |       0.8889 |      0.8889 |

환경별 결과에서는 BrightQuiet 환경에서 가장 높은 성능이 나타났고, DimFan 환경에서 가장 낮은 성능이 나타났다. 이는 저조도와 배경 소음이 동시에 존재하는 환경이 멀티모달 인식 시스템에 가장 어려운 조건임을 보여준다.

---

## 🎮 Unity 실시간 데모

본 프로젝트에서는 Python 기반 인식 시스템과 Unity를 UDP 통신으로 연동하여 실시간 캐릭터 제어 데모를 구현하였다.

### UDP 명령 형식

손 제스처 기반 이동 명령:

* `G:FORWARD`
* `G:BACKWARD`
* `G:LEFT`
* `G:RIGHT`
* `G:STOP`

음성 기반 행동 명령:

* `V:ATTACK`
* `V:DEFEND`

### Unity 구현 내용

Unity에서는 `UDPReceiver.cs`가 Python에서 전송한 UDP 문자열을 수신하고, `CommandController.cs`가 이를 해석하여 캐릭터 제어에 반영한다.

최종 Unity 데모에는 다음 기능을 적용하였다.

* Rider 캐릭터 모델 적용
* 전진, 후진, 좌우 이동 애니메이션 적용
* 공격 애니메이션 적용
* 방어 애니메이션 적용
* 3인칭 카메라 구현
* Root Motion 비활성화를 통한 안정적인 이동 처리
* UDP 명령을 한 번 처리한 뒤 비우는 방식으로 Attack / Defend 반복 실행 문제 해결

초기 구현에서는 캐릭터 상태를 색상 변화로 표현하였으나, 최종 구현에서는 Animator Controller를 이용하여 실제 공격 및 방어 애니메이션이 출력되도록 개선하였다.

---

## ⚙️ 저장소 구성

```text
hand_voice_fusion_project/
│
├── fusion_project/
│   ├── labels.csv              # 전체 데이터 라벨 정보
│   ├── splits/                 # train / val / test 분할 파일
│   ├── src/                    # 특징 추출, 학습, 평가 코드
│   └── outputs/                # 실험 결과 리포트 및 그래프
│
├── Unity/
│   ├── Assets/                 # Unity 씬, 스크립트, 모델, 애니메이션
│   ├── Packages/               # Unity 패키지 설정
│   └── ProjectSettings/        # Unity 프로젝트 설정
│
├── run_both.py                 # 손 제스처 + 음성 인식 동시 실행
├── send_hand_cmd_latefusion.py # 손 제스처 명령 전송
├── send_voice_cmd_latefusion.py# 음성 명령 전송
├── hand_env.yml                # Conda 환경 설정
├── conda.txt                   # 환경 관련 참고 파일
└── README.md
```

---

## ▶️ 실행 방법

### 1. Python 환경 설정

```bash
conda env create -f hand_env.yml
conda activate hand_env
```

### 2. 손 제스처 및 음성 명령 실행

```bash
python run_both.py
```

또는 각각 따로 실행할 수 있다.

```bash
python send_hand_cmd_latefusion.py
python send_voice_cmd_latefusion.py
```

### 3. Unity 데모 실행

Unity에서 `Unity/` 폴더를 프로젝트로 열고 `SampleScene`을 실행한다.
Python 스크립트가 UDP 명령을 전송하면 Unity 캐릭터가 해당 명령에 따라 이동, 공격, 방어 동작을 수행한다.

---

## ✔️ 결론

본 연구를 통해 손 제스처와 음성을 결합한 멀티모달 실시간 제어 시스템이 실제 인터랙션 환경에서 활용 가능함을 확인하였다. Early Fusion과 Late Fusion 모두 높은 이동 정확도와 안정적인 C_fusion CSR을 보였으며, 최종 실험에서는 Late Fusion이 행동 정확도와 전체 평균 성능 측면에서 소폭 우수한 결과를 나타냈다.

또한 Unity 데모를 통해 인식 결과가 실제 캐릭터 이동 및 행동 애니메이션으로 연결되는 과정을 구현함으로써, 멀티모달 HCI 시스템의 실제 적용 가능성을 확인하였다.

---

## Author

LI HAISONG
Department of Software Convergence
