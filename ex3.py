from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

YOLO_CONFIG_DIR = Path(__file__).resolve().parent / ".ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "best260408.pt"
DEFAULT_IMAGE = SCRIPT_DIR / "ex1-26.png"
DEFAULT_IMAGE_URL = (
    "https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/images/ex1-26.png"
)


def download_image(url: str, output_path: Path) -> Path:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://cs.kwansei.ac.jp/~kitamura/lecture/RyoikiJisshu/",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
    except Exception as exc:
        raise RuntimeError(f"Could not download image from {url}: {exc}") from exc

    if not content_type.lower().startswith("image/"):
        raise RuntimeError(
            f"Downloaded data is not an image. Content-Type was '{content_type}'."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return output_path


def detection_is_helmet(class_name: str, use_class_filter: bool) -> bool:
    if not use_class_filter:
        return True
    normalized = class_name.lower().replace("_", " ").replace("-", " ")
    helmet_words = ("helmet", "hardhat", "hard hat")
    return any(word in normalized for word in helmet_words)


def clamp_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def draw_count(image_bgr: np.ndarray, count: int) -> None:
    text = f"Blue helmets: {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 20, 40
    padding = 10

    cv2.rectangle(
        image_bgr,
        (x - padding, y - text_size[1] - padding),
        (x + text_size[0] + padding, y + baseline + padding),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(
        image_bgr,
        (x - padding, y - text_size[1] - padding),
        (x + text_size[0] + padding, y + baseline + padding),
        (0, 0, 255),
        2,
    )
    cv2.putText(
        image_bgr,
        text,
        (x, y),
        font,
        scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )


def show_result(image_bgr: np.ndarray, title: str = "Blue helmet detection") -> None:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)
    cv2.imshow(title, image_bgr)
    print("Press any key in the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def center_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax = (box_a[0] + box_a[2]) / 2
    ay = (box_a[1] + box_a[3]) / 2
    bx = (box_b[0] + box_b[2]) / 2
    by = (box_b[1] + box_b[3]) / 2
    return float(np.hypot(ax - bx, ay - by))


def remove_duplicate_boxes(
    candidates: list[tuple[float, tuple[int, int, int, int]]],
) -> list[tuple[float, tuple[int, int, int, int]]]:
    kept: list[tuple[float, tuple[int, int, int, int]]] = []

    for conf, box in sorted(candidates, key=lambda item: item[0], reverse=True):
        x1, y1, x2, y2 = box
        box_size = max(x2 - x1, y2 - y1)
        is_duplicate = False

        for _, kept_box in kept:
            kx1, ky1, kx2, ky2 = kept_box
            kept_size = max(kx2 - kx1, ky2 - ky1)
            close_enough = center_distance(box, kept_box) <= max(box_size, kept_size) * 0.8

            if close_enough:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append((conf, box))

    return kept


def run_detection(
    model_path: Path,
    image_path: Path,
    conf: float,
    imgsz: int,
    max_det: int,
) -> int:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else {}
    class_list = ", ".join(str(name) for name in names.values()) or "unknown"
    print(f"Detectable classes in this model: {class_list}")

    use_class_filter = any(
        detection_is_helmet(str(name), True) for name in names.values()
    )

    result = model.predict(
        str(image_path),
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )[0]
    annotated = image_bgr.copy()
    height, width = image_bgr.shape[:2]

    found = 0

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        draw_count(annotated, found)
        show_result(annotated)
        return 0

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for i, raw_box in enumerate(xyxy):
        class_id = int(cls_ids[i]) if cls_ids is not None else -1
        class_name = str(names.get(class_id, class_id))
        if not detection_is_helmet(class_name, use_class_filter):
            continue

        box = clamp_box(raw_box, width, height)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue

        candidates.append((float(confs[i]), box))

    for _, box in remove_duplicate_boxes(candidates):
        x1, y1, x2, y2 = box
        found += 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"blue helmet {found}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    draw_count(annotated, found)
    show_result(annotated)
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract blue helmet regions and draw red boxes plus the count."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to YOLO .pt model. Default: best260408.pt next to this script.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an input image. Default: ex1-26.png next to this script.",
    )
    parser.add_argument(
        "--image-url",
        default=DEFAULT_IMAGE_URL,
        help="Image URL used when --image is omitted.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="YOLO confidence. Increase this to reduce false detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO input image size. Larger default helps detect distant helmets.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Maximum number of helmet detections to draw.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"Model file was not found: {model_path}", file=sys.stderr)
        return 1

    if args.image:
        image_path = Path(args.image)
    elif DEFAULT_IMAGE.exists():
        image_path = DEFAULT_IMAGE
    else:
        image_path = SCRIPT_DIR / "ex1-26.png"
        try:
            download_image(args.image_url, image_path)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            print(
                "Download failed. Save ex1-26.png locally and run again with "
                "--image path\\to\\ex1-26.png",
                file=sys.stderr,
            )
            return 1

    count = run_detection(
        model_path=model_path,
        image_path=image_path,
        conf=args.conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
    )

    print(f"Blue helmet count: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())