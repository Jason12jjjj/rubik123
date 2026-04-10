"""
YOLOv8 Object Detection — Rubik's Cube Inference Module
=========================================================
Provides functions to detect Rubik's cube faces and individual stickers
using a trained YOLOv8 model (best.pt).

Main API:
    get_cube_bbox(image_path)  → dict with bbox, cropped image, confidence
    detect_stickers(image_path) → list of 9 sticker bboxes with colors

Usage:
    from yolo_detect import get_cube_bbox, detect_stickers

    result = get_cube_bbox("photo.jpg")
    print(result["bbox"])       # (x1, y1, x2, y2)
    print(result["cropped"])    # numpy array (BGR)
    print(result["confidence"]) # 0.0 – 1.0
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the trained YOLOv8 weights — update this after training
MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "runs", "detect", "rubik_cube", "weights", "best.pt"),
)

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD        = 0.45
IMG_SIZE             = 640

# Class names expected from the model
# Adapt these to match the Roboflow dataset labels
CLASS_CUBE     = "rubik-cube"       # full cube face class
CLASS_STICKER  = "sticker"          # individual sticker class
# If your dataset has colour-specific sticker classes, list them here:
COLOR_CLASSES  = {
    "white-sticker":  "White",
    "red-sticker":    "Red",
    "green-sticker":  "Green",
    "yellow-sticker": "Yellow",
    "orange-sticker": "Orange",
    "blue-sticker":   "Blue",
    # Fallbacks for single-class sticker models
    "sticker":        None,   # colour determined post-detection
    "rubik-cube":     None,
}

# ---------------------------------------------------------------------------
# Lazy model loader (singleton)
# ---------------------------------------------------------------------------

_model = None


def _load_model(model_path: str = None):
    """Load the YOLOv8 model once, reuse for all subsequent calls."""
    global _model
    if _model is not None:
        return _model

    path = model_path or MODEL_PATH

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"YOLOv8 model not found at: {path}\n"
            f"Please train a model first using train_yolo.py, or set the\n"
            f"YOLO_MODEL_PATH environment variable to your best.pt location."
        )

    from ultralytics import YOLO
    _model = YOLO(path)
    return _model


# ---------------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------------

def get_cube_bbox(image_input, model_path: str = None, draw: bool = False):
    """
    Detect the Rubik's cube face in an image and return its bounding box.

    Parameters
    ----------
    image_input : str | np.ndarray | bytes
        Filepath, BGR numpy array, or raw image bytes.
    model_path : str, optional
        Override path to the .pt weights file.
    draw : bool
        If True, return an annotated image with the bbox drawn on it.

    Returns
    -------
    dict or None
        {
            "bbox":       (x1, y1, x2, y2),    # pixel coordinates
            "center":     (cx, cy),              # centre point
            "width":      int,
            "height":     int,
            "confidence": float,                 # 0.0 – 1.0
            "cropped":    np.ndarray,            # cropped BGR region
            "annotated":  np.ndarray | None,     # only if draw=True
            "class_name": str,
        }
        Returns None if no cube is detected.
    """
    model = _load_model(model_path)
    img   = _read_image(image_input)

    results = model.predict(
        source=img,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    if not results or len(results[0].boxes) == 0:
        return None

    # Pick the detection with the highest confidence
    boxes = results[0].boxes
    best_idx   = int(boxes.conf.argmax())
    best_box   = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    best_conf  = float(boxes.conf[best_idx].cpu())
    best_cls   = int(boxes.cls[best_idx].cpu())
    class_name = model.names.get(best_cls, "unknown")

    x1, y1, x2, y2 = best_box
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    cropped = img[y1:y2, x1:x2].copy()

    annotated = None
    if draw:
        annotated = img.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{class_name} {best_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return {
        "bbox":       (x1, y1, x2, y2),
        "center":     ((x1 + x2) // 2, (y1 + y2) // 2),
        "width":      x2 - x1,
        "height":     y2 - y1,
        "confidence": best_conf,
        "cropped":    cropped,
        "annotated":  annotated,
        "class_name": class_name,
    }


def detect_stickers(image_input, model_path: str = None):
    """
    Detect individual stickers (up to 9) on a Rubik's cube face.

    Parameters
    ----------
    image_input : str | np.ndarray | bytes
        Image source (file path, BGR array, or raw bytes).

    Returns
    -------
    list[dict]
        Sorted list of sticker detections (top-left → bottom-right), each:
        {
            "bbox":       (x1, y1, x2, y2),
            "center":     (cx, cy),
            "confidence": float,
            "class_name": str,
            "color":      str | None,   # mapped colour name if available
            "cropped":    np.ndarray,
        }
    """
    model = _load_model(model_path)
    img   = _read_image(image_input)

    results = model.predict(
        source=img,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    if not results or len(results[0].boxes) == 0:
        return []

    boxes    = results[0].boxes
    h_img, w_img = img.shape[:2]
    stickers = []

    for i in range(len(boxes)):
        xyxy  = boxes.xyxy[i].cpu().numpy().astype(int)
        conf  = float(boxes.conf[i].cpu())
        cls   = int(boxes.cls[i].cpu())
        cname = model.names.get(cls, "unknown")

        x1, y1, x2, y2 = xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # Map class name to a Rubik's color if possible
        mapped_color = COLOR_CLASSES.get(cname, None)

        stickers.append({
            "bbox":       (x1, y1, x2, y2),
            "center":     ((x1 + x2) // 2, (y1 + y2) // 2),
            "confidence": conf,
            "class_name": cname,
            "color":      mapped_color,
            "cropped":    img[y1:y2, x1:x2].copy(),
        })

    # Sort by grid position: top → bottom, left → right
    stickers = _sort_as_grid(stickers, expected=9)
    return stickers


def detect_and_draw(image_input, model_path: str = None):
    """
    Run detection and return the fully annotated image (with all boxes drawn).

    Returns
    -------
    tuple[np.ndarray, list[dict]]
        (annotated_bgr_image, list_of_detections)
    """
    model = _load_model(model_path)
    img   = _read_image(image_input)

    results = model.predict(
        source=img,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    annotated = results[0].plot() if results else img.copy()

    detections = []
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy  = boxes.xyxy[i].cpu().numpy().astype(int)
            conf  = float(boxes.conf[i].cpu())
            cls   = int(boxes.cls[i].cpu())
            cname = model.names.get(cls, "unknown")
            detections.append({
                "bbox":       tuple(xyxy),
                "confidence": conf,
                "class_name": cname,
            })

    return annotated, detections


# ---------------------------------------------------------------------------
# Helper: crop cube and extract 9 sticker grid colours
# ---------------------------------------------------------------------------

def get_face_colors_from_crop(cropped_bgr, classifier_fn=None):
    """
    Given a cropped cube-face image, divide into a 3×3 grid and classify
    each sticker colour.

    Parameters
    ----------
    cropped_bgr : np.ndarray
        BGR image of the cube face only.
    classifier_fn : callable, optional
        Function(bgr_pixel) → str.  Falls back to basic HSV classification
        if not provided.

    Returns
    -------
    list[str]
        9 colour names in row-major order.
    """
    if classifier_fn is None:
        classifier_fn = _simple_hsv_classify

    h, w = cropped_bgr.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    colours = []

    for row in range(3):
        for col in range(3):
            cy = int((row + 0.5) * cell_h)
            cx = int((col + 0.5) * cell_w)
            # Sample a small patch around the centre of each cell
            pad = min(cell_h, cell_w) // 6
            patch = cropped_bgr[
                max(0, cy - pad): min(h, cy + pad),
                max(0, cx - pad): min(w, cx + pad),
            ]
            if patch.size == 0:
                colours.append("White")
                continue
            median_bgr = np.median(patch, axis=(0, 1)).astype(np.uint8)
            colours.append(classifier_fn(median_bgr))

    return colours


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_image(image_input) -> np.ndarray:
    """Normalise various input formats into a BGR numpy array."""
    if isinstance(image_input, np.ndarray):
        return image_input
    if isinstance(image_input, (bytes, bytearray)):
        arr = np.frombuffer(image_input, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image from bytes.")
        return img
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_input}")
        return img
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def _sort_as_grid(detections: list, expected: int = 9) -> list:
    """
    Sort detections into a 3×3 grid order (top-left → bottom-right)
    by clustering Y-coordinates into 3 rows, then sorting within each row by X.
    """
    if len(detections) < 2:
        return detections

    # Sort by Y centre first
    detections.sort(key=lambda d: d["center"][1])

    rows_n = int(np.sqrt(expected))  # 3 for a 3×3 grid
    chunk  = max(1, len(detections) // rows_n)

    sorted_dets = []
    for i in range(0, len(detections), chunk):
        row = detections[i: i + chunk]
        row.sort(key=lambda d: d["center"][0])  # sort by X within row
        sorted_dets.extend(row)

    return sorted_dets


def _simple_hsv_classify(bgr_pixel):
    """Minimal HSV colour classifier (fallback when rubiks_core is unavailable)."""
    px  = np.uint8([[bgr_pixel]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if s < 50 and v > 170:
        return "White"
    if (h < 10 or h > 165) and s > 100:
        return "Red" if v < 200 else "Orange"
    if 11 <= h <= 25 and s > 100:
        return "Orange"
    if 26 <= h <= 34:
        return "Yellow"
    if 35 <= h <= 85:
        return "Green"
    if 86 <= h <= 130:
        return "Blue"
    return "White"


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="YOLOv8 Rubik's Cube Detector")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--model", default=None, help="Path to best.pt weights")
    parser.add_argument("--mode", choices=["cube", "stickers", "draw"], default="cube",
                        help="Detection mode: cube (bounding box), stickers (9-grid), draw (annotated)")
    parser.add_argument("--output", default=None, help="Save annotated image to this path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf

    try:
        if args.mode == "cube":
            result = get_cube_bbox(args.image, model_path=args.model, draw=True)
            if result is None:
                print("❌ No cube detected.")
                sys.exit(1)
            print(f"✅ Cube detected!")
            print(f"   BBox:       {result['bbox']}")
            print(f"   Centre:     {result['center']}")
            print(f"   Size:       {result['width']}×{result['height']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Class:      {result['class_name']}")

            if args.output and result["annotated"] is not None:
                cv2.imwrite(args.output, result["annotated"])
                print(f"   Saved → {args.output}")

        elif args.mode == "stickers":
            stickers = detect_stickers(args.image, model_path=args.model)
            if not stickers:
                print("❌ No stickers detected.")
                sys.exit(1)
            print(f"✅ {len(stickers)} sticker(s) detected:")
            for i, s in enumerate(stickers):
                print(f"   [{i}] bbox={s['bbox']}  conf={s['confidence']:.2f}  "
                      f"class={s['class_name']}  color={s['color']}")

        elif args.mode == "draw":
            annotated, dets = detect_and_draw(args.image, model_path=args.model)
            print(f"✅ {len(dets)} detection(s).")
            out = args.output or "yolo_result.jpg"
            cv2.imwrite(out, annotated)
            print(f"   Saved → {out}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
