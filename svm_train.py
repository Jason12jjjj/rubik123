import os
import cv2
import numpy as np
from sklearn.svm import SVC
import joblib

# ==========================================
# 路径设置 (已优化为本地路径)
# ==========================================
dataset_path = "svm_dataset"
model_save_path = "svm_color_model.pkl"

def extract_features(image_path):
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def train_model():
    print("=== 开始训练魔方颜色 SVM 模型 ===")
    features, labels = [], []
    
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 找不到数据集文件夹 {dataset_path}")
        return

    for color_name in os.listdir(dataset_path):
        color_dir = os.path.join(dataset_path, color_name)
        if not os.path.isdir(color_dir): continue
            
        print(f"正在读取类别: {color_name} ...")
        for img_name in os.listdir(color_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            img_path = os.path.join(color_dir, img_name)
            feature = extract_features(img_path)
            if feature is not None:
                features.append(feature)
                labels.append(color_name)
                
    if not features:
        print("⚠️ 未找到有效图片数据！")
        return

    print("数据加载完毕，开始训练 SVM ...")
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(features, labels)
    joblib.dump(svm_model, model_save_path)
    print(f"✅ 训练完成！模型已保存至: {model_save_path}")

if __name__ == "__main__":
    train_model()
