"""
Rubik's YOLOv8 Synthetic Trainer
This script generates a synthetic dataset of Rubik's cube faces and trains a YOLOv8 model.
Designed for Google Colab.
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path

# --- Configuration ---
DATASET_PATH = Path("rubiks_dataset")
IMG_SIZE = 640
TRAIN_COUNT = 500  # Number of synthetic images to generate
VAL_COUNT = 100
CLASSES = ['cube', 'sticker']

def create_synthetic_cube():
    """Generates a synthetic Rubik's face and returns the image and YOLO labels."""
    # Background
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img[:] = [random.randint(50, 150) for _ in range(3)] # Random dark background

    # Random noise/gradient
    for _ in range(10):
        cv2.circle(img, (random.randint(0,640), random.randint(0,640)), 
                   random.randint(100, 300), 
                   [random.randint(0,255) for _ in range(3)], -1)
    img = cv2.GaussianBlur(img, (99, 99), 0)

    # Cube Parameters
    cube_size = random.randint(200, 400)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    
    # Square colors
    colors = [
        (255, 255, 255), # White
        (0, 0, 255),     # Red (BGR)
        (0, 255, 0),     # Green
        (0, 255, 255),   # Yellow
        (0, 165, 255),   # Orange
        (255, 0, 0)      # Blue
    ]
    
    # Draw logic
    s = cube_size // 3
    padding = s // 10
    
    # Cube BBox calculation
    cube_x1, cube_y1 = cx - cube_size//2, cy - cube_size//2
    cube_x2, cube_y2 = cx + cube_size//2, cy + cube_size//2
    
    labels = []
    # Class 0: Cube
    labels.append(f"0 {(cx/640):.6f} {(cy/640):.6f} {(cube_size/640):.6f} {(cube_size/640):.6f}")

    for r in range(3):
        for c in range(3):
            color = random.choice(colors)
            x1 = cube_x1 + c * s + padding
            y1 = cube_y1 + r * s + padding
            x2 = x1 + s - 2*padding
            y2 = y1 + s - 2*padding
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), 2) # black border
            
            # Class 1: Sticker
            sx_c, sy_c = (x1+x2)/2, (y1+y2)/2
            sw, sh = (x2-x1), (y2-y1)
            labels.append(f"1 {(sx_c/640):.6f} {(sy_c/640):.6f} {(sw/640):.6f} {(sh/640):.6f}")

    # Random Rotation
    angle = random.randint(-30, 30)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
    
    # Note: For simplicity in this demo, we won't rotate the label bboxes.
    # In a real training, we'd use Albumentations or similar.
    
    return img, labels

def setup_dirs():
    """Create directory structure for YOLO training."""
    for split in ['train', 'val']:
        (DATASET_PATH / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_PATH / 'labels' / split).mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH / 'data.yaml', 'w') as f:
        f.write(f"path: {DATASET_PATH.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"names:\n  0: {CLASSES[0]}\n  1: {CLASSES[1]}\n")

def generate_dataset():
    """Generates the full dataset files."""
    print("🚀 Generating synthetic dataset...")
    setup_dirs()
    
    for split, count in [('train', TRAIN_COUNT), ('val', VAL_COUNT)]:
        for i in range(count):
            img, labels = create_synthetic_cube()
            fname = f"{split}_{i}"
            cv2.imwrite(str(DATASET_PATH / 'images' / split / f"{fname}.jpg"), img)
            with open(DATASET_PATH / 'labels' / split / f"{fname}.txt", 'w') as f:
                f.write("\n".join(labels))
    print("✅ Dataset ready.")

def train_model():
    """Installs ultralytics and performs training."""
    print("📦 Installing Ultralytics...")
    os.system("pip install ultralytics")
    
    from ultralytics import YOLO
    
    print("🛠️ Initializing Training...")
    # Standardize: Start from best.pt if exists, otherwise start from yolov8n.pt
    base_model = 'best.pt' if os.path.exists('best.pt') else 'yolov8n.pt'
    model = YOLO(base_model) 
    
    results = model.train(
        data=str(DATASET_PATH / 'data.yaml'),
        epochs=50,
        imgsz=640,
        device=0, # GPU
        project='rubiks_solver',
        name='v1'
    )
    print("🎉 Training Complete!")
    print(f"Model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    generate_dataset()
    # If running in Colab, uncomment the next line to start training
    # train_model()
