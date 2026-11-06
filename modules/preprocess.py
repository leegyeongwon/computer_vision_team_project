"""
preprocess.py
-------------
화이트밸런스(Gray World) -> 감마 -> CLAHE(L 채널) 전처리.

- wb: 색감 조절
- gamma: 밝기 조절
- 디테일 살리기

- WB 게인과 감마는 빈 배경 프레임으로 한 번 보정(calibrate)해 고정한다.
- CLAHE만 매 프레임 적용한다.
- apply()의 wb / gamma / clahe 플래그로 단계별 on/off가 가능하다. (실험용)
"""

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, target_mean=110.0, clip_limit=2.0, tile=(8, 8), kernel_size=(5, 5), sigma=0.0):
        self.target_mean = target_mean
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
        self.kernel_size = kernel_size  # 가우시안 필터 커널 크기 (예: (3,3), (5,5))
        self.sigma = sigma              # 가우시안 표준편차 (0이면 자동 계산)
        
        self.gains = np.ones(3, np.float32)         # B, G, R
        self.gamma = 1.0
        self.lut = np.arange(256, dtype=np.uint8)
        self.calibrated = False

    def calibrate(self, bg_frame, verbose=True):
        """사람이 없는 배경 프레임(BGR)으로 게인과 감마를 계산한다."""
        means = bg_frame.reshape(-1, 3).mean(axis=0)                 # B, G, R 평균
        gains = means.mean() / np.maximum(means, 1e-6)
        self.gains = np.clip(gains, 0.6, 1.6).astype(np.float32)    # 과보정 방지

        balanced = self._white_balance(bg_frame)
        mean = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY).mean() / 255.0
        mean = float(np.clip(mean, 0.02, 0.98))
        target = self.target_mean / 255.0
        self.gamma = float(np.clip(np.log(target) / np.log(mean), 0.5, 2.0))

        x = np.arange(256) / 255.0
        self.lut = np.clip((x ** self.gamma) * 255.0, 0, 255).astype(np.uint8)
        self.calibrated = True
        if verbose:
            print(f"Preprocess calibrated: gains(BGR)={self.gains.round(2)}, "
                  f"gamma={self.gamma:.2f}")

    def _white_balance(self, img):
        return np.clip(img.astype(np.float32) * self.gains, 0, 255).astype(np.uint8)

    def apply(self, img, blur=True, wb=True, gamma=True, clahe=True):
        """켜진 단계만 적용한다. 모두 꺼져 있으면 입력을 그대로 돌려준다."""
        out = img
        # 1. 가우시안 필터 (노이즈 완화)
        if blur:
            out = cv2.GaussianBlur(out, self.kernel_size, self.sigma)
        # 2. 화이트 밸런스
        if wb:
            out = self._white_balance(out)
        # 3. 감마 보정
        if gamma:
            out = cv2.LUT(out, self.lut)
        # 4. CLAHE (대비 개선)
        if clahe:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        return out