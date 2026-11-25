import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = SCRIPT_DIR / "ex5-26.mp4"
MODEL_PATH = SCRIPT_DIR / "best260408.pt"

CONFIDNTH = 0.2  # 誤検出が多い場合は 0.10 や 0.15 を試す
IMAGE_SIZE = 1280
REPORT_INTERVAL = 30
WINDOW_NAME = "Blue Helmet Detection"

# 青いヘルメットのクラスIDを明示する場合は、None を整数に変更する。
BLUE_HELMET_CLASS_ID = None


def get_blue_helmet_class_id(names: dict[int, str]) -> int:
    if BLUE_HELMET_CLASS_ID is not None:
        if BLUE_HELMET_CLASS_ID not in names:
            raise SystemExit(
                f"指定したクラスIDがモデルにありません。モデルのクラス {names}"
            )
        return BLUE_HELMET_CLASS_ID

    # 青いヘルメット専用の1クラスモデルでは、そのクラスを使う。
    if len(names) == 1:
        return next(iter(names))

    # 複数クラスの場合は、クラス名から青いヘルメットを特定する。
    matches = []
    for class_id, name in names.items():
        normalized = name.lower().replace("_", " ").replace("-", " ")
        words = normalized.split()
        is_helmet = (
            "helmet" in words
            or "hardhat" in words
            or "hard hat" in normalized
        )
        if "blue" in words and is_helmet:
            matches.append(class_id)

    if len(matches) == 1:
        return matches[0]

    raise SystemExit(
        f"青いヘルメットのクラスIDを特定できません。モデルのクラス {names}\n"
        "該当するIDをコード先頭の BLUE_HELMET_CLASS_ID に設定してください。"
    )


def main() -> None:
    if not 0.0 <= CONFIDNTH <= 1.0:
        raise SystemExit("CONFIDNTH には 0.0 から 1.0 までの値を指定してください。")

    if not VIDEO_PATH.is_file():
        raise SystemExit(f"動画が見つかりません。 {VIDEO_PATH}")

    if not MODEL_PATH.is_file():
        raise SystemExit(f"モデルが見つかりません。 {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    class_id = get_blue_helmet_class_id(model.names)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"動画を開けません。 {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_duration = 1.0 / fps

    frame_number = 0
    window_created = False

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_created = True

        while True:
            started_at = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1

            # 青いヘルメットのクラスだけを検出する。
            result = model.predict(
                source=frame,
                classes=[class_id],
                conf=CONFIDNTH,
                imgsz=IMAGE_SIZE,
                verbose=False,
            )[0]

            boxes = result.boxes
            count = 0

            if boxes is not None:
                count = len(boxes)
                height, width = frame.shape[:2]
                coordinates = boxes.xyxy.round().int().cpu().tolist()

                for x1, y1, x2, y2 in coordinates:
                    left = max(0, min(x1, width - 1))
                    top = max(0, min(y1, height - 1))
                    right = max(0, min(x2, width - 1))
                    bottom = max(0, min(y2, height - 1))

                    # OpenCVの色指定はBGR順。 (0, 0, 255) は赤。
                    cv2.rectangle(
                        frame,
                        (left, top),
                        (right, bottom),
                        (0, 0, 255),
                        thickness=3,
                    )

            if frame_number % REPORT_INTERVAL == 0:
                print(
                    f"{frame_number} フレーム目: 青いヘルメット {count} 個",
                    flush=True,
                )

            cv2.imshow(WINDOW_NAME, frame)

            # 推論と描画にかかった時間を差し引いて待機する。
            elapsed = time.perf_counter() - started_at
            delay_ms = max(1, round((frame_duration - elapsed) * 1000))

            if cv2.waitKey(delay_ms) != -1:
                break

        if frame_number == 0:
            raise SystemExit("動画からフレームを読み取れませんでした。")

    finally:
        cap.release()
        if window_created:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()