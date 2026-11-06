import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from modules.preprocess import Preprocessor

photos = ["mountain.jpg"]
photo_num = 1

for img in photos:

    INPUT = img   # 입력 사진 경로

    img = cv2.imread(INPUT)
    pre = Preprocessor()
    pre.calibrate(img)    # 이 사진으로 WB 게인과 감마를 계산

    results = {
        "original": img,
        "wb": pre.apply(img, wb=True, gamma=False, clahe=False, blur=False),
        "gamma": pre.apply(img, wb=False, gamma=True, clahe=False, blur=False),
        "clahe": pre.apply(img, wb=False, gamma=False, clahe=True, blur=False),
        "all": pre.apply(img, wb=True, gamma=True, clahe=True, blur=False),
        "all + gausian": pre.apply(img, wb=True, gamma=True, clahe=True, blur=True)
    }

    for name, out in results.items():
        cv2.imwrite(f"example_image/photo{photo_num}_{name}.png", out)

    photo_num += 1