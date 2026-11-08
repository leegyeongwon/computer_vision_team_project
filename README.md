# 스쿼트 자세 검출 (MediaPipe Pose + MOG2 + 조명 전처리 실험)

웹캠으로 스쿼트 자세를 실시간 분석하는 프로젝트입니다.
MediaPipe Pose Landmarker로 관절을 검출하고, 조명 조건(bright / dark)에 따라
전처리(화이트밸런스, 감마, CLAHE) 유무가 검출에 미치는 영향을 실험·기록할 수 있습니다.

## 주요 기능

- **MediaPipe Pose (VIDEO 모드)**: 한 명의 자세를 추적하며 랜드마크 검출
- **MOG2 배경 제거 + ROI 크롭 (선택)**: 빈 배경을 학습한 뒤 전경 영역만 잘라 MediaPipe에 입력
- **조명 전처리 (MediaPipe 입력에만 적용)**: WB → 감마 → CLAHE
  - MOG2에는 항상 원본 프레임이 들어가므로 전처리의 영향을 받지 않음
  - WB/감마는 `b` 키로 배경을 고정하는 순간 빈 배경 기준으로 한 번만 보정하고, CLAHE만 매 프레임 적용
- **스쿼트 분석**: 관절 각도 계산, 자세 피드백, 오버레이 표시 (`modules/squat_analyzer.py`)
- **실험 기록**: 시행 단위로 녹화하고 CSV로 저장 (`modules/experiment_logger.py`)

## 폴더 구조

```
.
├── main.py
├── pose_landmarker_lite.task      # MediaPipe 모델 파일
└── modules/
    ├── squat_analyzer.py          # 각도 계산, 피드백, 오버레이
    ├── preprocess.py              # Preprocessor (WB / 감마 / CLAHE)
    └── experiment_logger.py       # ExperimentLogger (CSV 저장)
```

`pose_landmarker_lite.task`는 `main.py`와 **같은 폴더**에 있어야 합니다.
MediaPipe 공식 문서의 Pose Landmarker 모델 목록에서 `pose_landmarker_lite`를 받을 수 있습니다.

## 설치

Python 3.9 이상을 권장합니다.

```bash
pip install opencv-python mediapipe numpy
```

> `modules/` 안의 파일이 다른 패키지를 쓴다면 함께 설치해야 합니다.

## 실행

```bash
python main.py
```

`main.py`만 실행하면 됩니다. 단, 아래 조건이 필요합니다.

1. 웹캠이 연결되어 있고 카메라 권한이 허용되어 있을 것 (기본 카메라 `0` 사용)
2. `pose_landmarker_lite.task`가 `main.py`와 같은 폴더에 있을 것
3. `modules/` 폴더가 `main.py`와 같은 위치에 있을 것

## 사용 순서

1. 실행하면 **MOG2 Foreground Mask** 창이 뜨고 빈 배경을 학습합니다.
2. **카메라 화면 밖으로 나가서** 배경만 보이게 한 뒤 `b`를 눌러 배경을 고정합니다.
   이때 WB/감마 보정값도 함께 계산됩니다.
3. 카메라 앞으로 들어오면 **MediaPipe Pose Detection** 창에서 자세 검출이 시작됩니다.
   (MediaPipe에 실제로 들어가는 이미지는 **MediaPipe Input** 창에서 확인)
4. `m`, `p`, `l` 키로 입력 방식, 전처리 조건, 조명 라벨을 정하고 `e`로 시행을 녹화합니다.
5. `q`로 종료하면 녹화 중이던 시행도 저장됩니다.

카메라는 측면·고정 상태로 전신이 보이게 두는 것을 전제로 합니다.

## 키 조작

| 키 | 동작 |
|----|------|
| `b` | 배경 고정 (전처리 보정 포함) |
| `m` | 입력 방식 전환: Original ↔ MOG2 ROI |
| `p` | 전처리 조건 순환 |
| `l` | 조명 라벨 순환 |
| `e` | 시행 녹화 시작 / 종료 |
| `r` | 리셋 (배경 재학습 단계로 복귀) |
| `s` | 분석 패널 표시 토글 |
| `x` | 새 세트 시작 |
| `q` | 종료 |

녹화 중(`e`)에는 `m`, `p`, `l`로 설정을 바꿀 수 없습니다. 먼저 녹화를 종료하세요.
`m`, `p`, `e`는 배경을 고정(`b`)한 뒤에만 동작합니다.

## 전처리 조건 (`p`)

| 조건 | 적용 내용 |
|------|-----------|
| OFF | 기본 블러만 적용 |
| WB | 블러 + 화이트밸런스 |
| GAMMA | 블러 + 감마 보정 |
| CLAHE | 블러 + CLAHE |
| WB+GAMMA | 블러 + 화이트밸런스 + 감마 |
| ALL | 블러 + WB + 감마 + CLAHE |

조명 라벨(`l`)은 `bright`, `dark`, `backlit`, `sidelit` 순으로 순환하며 기록에 함께 저장됩니다.

## 실험 기록

- `e`를 누르면 현재 전처리 조건과 조명 라벨로 한 시행의 녹화가 시작되고, 다시 누르면 CSV로 저장됩니다.
- 저장 위치와 컬럼 구성: **TODO (experiment_logger.py 기준으로 작성)**

## 문제 해결

- `Cannot open webcam`: 카메라 권한과 연결을 확인하세요. 다른 앱이 카메라를 쓰고 있어도 실패할 수 있습니다.
- 모델 파일 오류: `pose_landmarker_lite.task`가 `main.py`와 같은 폴더에 있는지 확인하세요.
- `No pose detected`: 전신이 화면에 보이도록 거리를 조정하세요.
- 후면 카메라를 쓸 경우 `main.py`의 `cv2.flip(frame, 1)` 줄을 주석 처리하세요.
