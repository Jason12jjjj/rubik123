import os
import cv2
import time
import numpy as np

# ==========================================
# 1. 路径设置 (已优化为本地路径)
# ==========================================
result_folder = "predictions"   # 这里存放 predict 之后的结果图
raw_folder = "to_predict"       # 这里存放原始的原图
dataset_path = "svm_dataset"    # 修正后的数据将存入这里

# ==========================================
# 2. 鼠标点击与按钮交互逻辑
# ==========================================
# 记录鼠标点击坐标
click_pos = [None]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = (x, y)

def get_action_from_click(x, y):
    """根据鼠标点击的坐标，判断按下了哪个虚拟按钮"""
    if 610 <= y <= 660:
        if 20 <= x <= 100: return 'red'
        if 115 <= x <= 195: return 'green'
        if 210 <= x <= 290: return 'blue'
        if 305 <= x <= 385: return 'yellow'
        if 400 <= x <= 480: return 'orange'
        if 495 <= x <= 575: return 'white'
    elif 680 <= y <= 730:
        if 20 <= x <= 195: return 'skip'
        if 210 <= x <= 385: return 'next'
        if 400 <= x <= 575: return 'quit'
    return None

def draw_control_panel(canvas):
    """在图片底部绘制控制面板和按钮"""
    # 底部深灰色背景
    cv2.rectangle(canvas, (0, 600), (600, 750), (40, 40, 40), -1)

    # --- 第一排：颜色按钮 ---
    # 红 (BGR)
    cv2.rectangle(canvas, (20, 610), (100, 660), (0, 0, 255), -1)
    cv2.putText(canvas, "RED(r)", (35, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # 绿
    cv2.rectangle(canvas, (115, 610), (195, 660), (0, 200, 0), -1)
    cv2.putText(canvas, "GRN(g)", (125, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # 蓝
    cv2.rectangle(canvas, (210, 610), (290, 660), (255, 0, 0), -1)
    cv2.putText(canvas, "BLU(b)", (225, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # 黄
    cv2.rectangle(canvas, (305, 610), (385, 660), (0, 255, 255), -1)
    cv2.putText(canvas, "YLW(y)", (320, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    # 橙
    cv2.rectangle(canvas, (400, 610), (480, 660), (0, 140, 255), -1)
    cv2.putText(canvas, "ORG(o)", (415, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # 白
    cv2.rectangle(canvas, (495, 610), (575, 660), (255, 255, 255), -1)
    cv2.putText(canvas, "WHT(w)", (510, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # --- 第二排：功能按钮 ---
    cv2.rectangle(canvas, (20, 680), (195, 730), (100, 100, 100), -1)
    cv2.putText(canvas, "Skip Cell (Space)", (40, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.rectangle(canvas, (210, 680), (385, 730), (100, 100, 100), -1)
    cv2.putText(canvas, "Next Image (n)", (240, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.rectangle(canvas, (400, 680), (575, 730), (80, 80, 200), -1)
    cv2.putText(canvas, "Quit (q)", (460, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ==========================================
# 3. 主程序逻辑
# ==========================================
color_map = {
    ord('r'): 'red', ord('g'): 'green', ord('b'): 'blue',
    ord('y'): 'yellow', ord('o'): 'orange', ord('w'): 'white'
}

def interactive_correct():
    if not os.path.exists(result_folder):
        print(f"❌ 错误：找不到结果文件夹 {result_folder}")
        return

    result_files = [f for f in os.listdir(result_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not result_files:
        print("✅ 结果文件夹中没有找到图片。")
        return

    print("=== 🛠️ 全新图形化纠错器已启动 ===")
    print("👉 请直接用鼠标在弹出的窗口上点击操作！CMD 现已转为后台日志输出。")

    # 初始化 OpenCV 窗口并绑定鼠标事件
    cv2.namedWindow("Interactive Corrector UI")
    cv2.setMouseCallback("Interactive Corrector UI", mouse_callback, click_pos)

    for res_name in result_files:
        res_path = os.path.join(result_folder, res_name)
        res_img = cv2.imdecode(np.fromfile(res_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if res_img is None: continue
        
        # 溯源清洗逻辑（保持上一版的高效逻辑）
        if "_result" in res_name:
            base_name = res_name.rsplit("_result", 1)[0]
        else:
            base_name = os.path.splitext(res_name)[0]
            
        raw_path = None
        if os.path.exists(raw_folder):
            for file in os.listdir(raw_folder):
                if os.path.splitext(file)[0] == base_name:
                    raw_path = os.path.join(raw_folder, file)
                    break
        
        if raw_path is None or not os.path.exists(raw_path):
            print(f"⚠️ 跳过 {res_name}: 在 {raw_folder} 中找不到核心名为 '{base_name}' 的原图。")
            continue
            
        raw_img = cv2.imdecode(np.fromfile(raw_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw_img is None: continue
        
        h, w, _ = raw_img.shape
        gh, gw = h // 3, w // 3
        skip_image = False
        
        for i in range(3):
            if skip_image: break
            for j in range(3):
                # 建立 600x750 的画布 (上面600是图片，下面150是UI)
                canvas = np.zeros((750, 600, 3), dtype=np.uint8)
                
                # 调整原结果图尺寸为 600x600 并画红框
                img_600 = cv2.resize(res_img, (600, 600))
                cv2.rectangle(img_600, (j*200, i*200), ((j+1)*200, (i+1)*200), (0, 0, 255), 6)
                canvas[:600, :] = img_600
                
                # 绘制控制面板
                draw_control_panel(canvas)
                
                cv2.imshow("Interactive Corrector UI", canvas)
                click_pos[0] = None # 清空之前的点击记录
                
                # 事件监听循环 (每 10ms 监听一次鼠标或键盘)
                action = None
                while True:
                    key = cv2.waitKey(10) & 0xFF
                    # 优先检测键盘快捷键
                    if key != 255:
                        if key in color_map: action = color_map[key]
                        elif key == ord(' '): action = 'skip'
                        elif key == ord('n'): action = 'next'
                        elif key == ord('q'): action = 'quit'
                    
                    # 检测鼠标点击
                    if click_pos[0] is not None:
                        cx, cy = click_pos[0]
                        action = get_action_from_click(cx, cy)
                        click_pos[0] = None # 响应后立即清空
                    
                    # 如果有合法操作，跳出循环执行动作
                    if action: break
                
                # --- 执行 Action 动作 ---
                if action in ['red', 'green', 'blue', 'yellow', 'orange', 'white']:
                    # 截取原图纯色块入库
                    crop_y, crop_x = int(gh * 0.1), int(gw * 0.1)
                    roi = raw_img[i*gh + crop_y : (i+1)*gh - crop_y, 
                                  j*gw + crop_x : (j+1)*gw - crop_x]
                    
                    target_dir = os.path.join(dataset_path, action)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    save_name = f"fixed_cell_{i}_{j}_{int(time.time()*1000)}.jpg"
                    cv2.imencode('.jpg', roi)[1].tofile(os.path.join(target_dir, save_name))
                    print(f"✅ 操作已执行: 坐标 [{i+1},{j+1}] 已存入 [{action}] 数据集")
                    
                elif action == 'next': 
                    print("⏭️ 跳过当前图片剩余格子")
                    skip_image = True
                    break
                elif action == 'quit':
                    print("🚪 退出图形化纠错器...")
                    cv2.destroyAllWindows()
                    return
                elif action == 'skip':
                    continue # 不做任何保存，直接下一个格子

    cv2.destroyAllWindows()
    print("--- 🎉 纠错结束，建议重新运行 svm_train.py 以训练新数据！ ---")

if __name__ == "__main__":
    interactive_correct()
