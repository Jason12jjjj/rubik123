import os
import cv2
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 路径设置 (已优化为本地路径)
# ==========================================
model_path = "svm_color_model.pkl"
to_predict_path = "to_predict"
output_dir = "predictions"

def extract_features_from_block(img_block):
    hsv = cv2.cvtColor(img_block, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def predict_rubiks_face():
    # 确保文件夹存在
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(to_predict_path): os.makedirs(to_predict_path)
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}，请先运行 svm_train.py")
        return

    print("=== 开始 3x3 矩阵检测 (稳定审计版) ===")
    svm_model = joblib.load(model_path)
    classes = list(svm_model.classes_)
    
    images_to_predict = [f for f in os.listdir(to_predict_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images_to_predict:
        print(f"⚠️ {to_predict_path} 文件夹中没有找到图片。请放入需要预测的魔方照片。")
        return

    for img_name in images_to_predict:
        img_path = os.path.join(to_predict_path, img_name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        
        file_name = os.path.splitext(img_name)[0]
        h, w, _ = img.shape
        gh, gw = h // 3, w // 3
        
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, "RGBA") 
        
        try:
            font = ImageFont.truetype("arial.ttf", max(16, int(w/22)))
        except IOError:
            font = ImageFont.load_default()
            
        report_lines = [f"STABLE FULL AUDIT REPORT: {file_name}", "="*65, ""]
        
        for row in range(3):
            for col in range(3):
                y_start, y_end = row * gh, (row + 1) * gh
                x_start, x_end = col * gw, (col + 1) * gw
                
                cell_img = img[y_start:y_end, x_start:x_end]
                # 向内缩进10%提取纯净颜色
                crop_y, crop_x = int(gh * 0.1), int(gw * 0.1)
                center_cell = cell_img[crop_y:gh-crop_y, crop_x:gw-crop_x]
                
                feature = extract_features_from_block(center_cell).reshape(1, -1)
                
                # [核心逻辑修复] 完全依靠 predict_proba 来决定颜色，避免底层冲突
                probabilities = svm_model.predict_proba(feature)[0]
                max_index = np.argmax(probabilities)
                prediction = classes[max_index]
                max_prob = probabilities[max_index]
                
                # --- 写入 Audit TXT 报告 ---
                report_lines.append(f"▉ GRID [{row+1}, {col+1}] ANALYSIS:")
                report_lines.append(f"  - FINAL DECISION: {prediction} ({max_prob*100:.1f}%)")
                report_lines.append(f"    Status: ✅ MATCHES MODEL")
                
                report_lines.append(f"\n  - MODEL PROBABILITY DISTRIBUTION:")
                for k, label in enumerate(classes):
                    p = probabilities[k]
                    bar = "█" * int(p * 20)
                    report_lines.append(f"    {label.ljust(7)}: {p*100:5.2f}% {bar}")
                report_lines.append("-" * 65 + "\n")
                
                # --- 绘制 UI 标签 ---
                txt = f"{prediction} {max_prob*100:.0f}%"
                tx, ty = col * gw + 15, row * gh + 25
                
                try:
                    bbox = draw.textbbox((tx, ty), txt, font=font)
                    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=(0,0,0,180))
                except AttributeError:
                    draw.rectangle([tx-5, ty-5, tx+100, ty+30], fill=(0,0,0,180))
                draw.text((tx, ty), txt, font=font, fill=(255,255,255))
                
        # 绘制绿色网格
        final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        for k in range(4): 
            cv2.line(final_img, (0, k*gh), (w, k*gh), (0,255,0), 2)
            cv2.line(final_img, (k*gw, 0), (k*gw, h), (0,255,0), 2)

        # 保存结果图 (带 _result) 和 报告 (带 _audit)
        res_img_path = os.path.join(output_dir, f"{file_name}_result.jpg")
        cv2.imencode('.jpg', final_img)[1].tofile(res_img_path)
        
        txt_path = os.path.join(output_dir, f"{file_name}_audit.txt")
        with open(txt_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(report_lines))
            
        print(f"✅ 处理完成: {img_name} -> 结果已生成在 {output_dir} 文件夹中")

if __name__ == "__main__":
    predict_rubiks_face()
