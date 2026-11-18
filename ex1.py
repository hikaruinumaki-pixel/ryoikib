import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_PATH = "ex1-26.png"
CONFIDENCE = 0.11


def main() -> None:
    try:
        image_bytes = Path(IMAGE_PATH).read_bytes()
    except OSError as exc:
        raise SystemExit(f"入力画像を開けません。 {exc}") from exc

    if not image_bytes:
        raise SystemExit(f"入力画像が空です。 {IMAGE_PATH}")

    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise SystemExit(f"入力画像を読み込めません。 {IMAGE_PATH}")

    # COCOで学習済みのモデルでは、クラスID 0が人物（person）。
    model = YOLO("yolo11n.pt")
    result = model.predict(
        source=image,
        classes=[0],
        conf=CONFIDENCE,
        verbose=False,
    )[0]

    boxes = result.boxes.xyxy.cpu().tolist()
    annotated = image.copy()
    height, width = image.shape[:2]
    thickness = max(2, round(min(width, height) / 300))

    for x1, y1, x2, y2 in boxes:
        left = max(0, min(width - 1, math.floor(x1)))
        top = max(0, min(height - 1, math.floor(y1)))
        right = max(0, min(width - 1, math.ceil(x2)))
        bottom = max(0, min(height - 1, math.ceil(y2)))
        cv2.rectangle(
            annotated, (left, top), (right, bottom), (0, 0, 255), thickness
        )

    count = len(boxes)
    label = f"People: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(width, height) / 850)
    (_, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    position = (8, text_height + 8)

    # 白文字に黒い縁取りを付ける。
    cv2.putText(
        annotated, label, position, font, font_scale,
        (0, 0, 0), thickness + 2, cv2.LINE_AA,
    )
    cv2.putText(
        annotated, label, position, font, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )

    print(f"検出した人物の数は {count} 人です。")
    print("画像ウィンドウでキーを押すと終了します。")

    window_name = "Person Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()