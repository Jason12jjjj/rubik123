import os, json, io, base64
from collections import Counter
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from rubiks_core import solve_cube

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Rubik's AI Solver", page_icon="🧊",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# CSS & STYLE (Safe Area for Streamlit Menu)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    font-family:'Outfit',sans-serif!important;
    background: radial-gradient(circle at 0% 0%, #f8fafc 0%, #e2e8f0 100%)!important;
}
[data-testid="stHeader"] { background-color: transparent !important; }
[data-testid="stMainBlockContainer"]{ padding-top: 50px !important; }
.app-title { font-size: 2.8rem; font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; }
.app-subtitle { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; }
.mcard{
    background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 28px; padding: 32px;
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.05);
}
.stButton>button{ border-radius: 12px!important; font-weight: 800!important; transition: all 0.2s; }
.stButton>button:hover{ transform: translateY(-2px); border-color: #6366f1!important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
FACES = ['Up','Left','Front','Right','Back','Down']
HEX_COLORS = {'White':'#f1f5f9','Red':'#ef4444','Green':'#22c55e', 'Yellow':'#eab308','Orange':'#f97316','Blue':'#3b82f6'}
COLOR_EMOJIS = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green', 'Right':'Red','Back':'Blue','Down':'Yellow'}
CALIB_FILE = "calibration_profile.json"

_DEFAULTS = {
    'active_face': 'Front',
    'cube_state': {f: (['White']*4+[CENTER_COLORS[f]]+['White']*4) for f in FACES},
    'last_solution': None,
    'custom_std_colors': {},
    'history': None,
    'history_index': 0,
    'confirmed_faces': [],
    'scan_result': None,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

if st.session_state.history is None:
    st.session_state.history = [json.dumps({"cube_state": st.session_state.cube_state, "confirmed_faces": st.session_state.confirmed_faces})]

def push_history():
    sj = json.dumps({"cube_state": st.session_state.cube_state, "confirmed_faces": st.session_state.confirmed_faces})
    st.session_state.history.append(sj)
    st.session_state.history_index = len(st.session_state.history) - 1

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS & VISION ENGINES
# ══════════════════════════════════════════════════════════════════════════════
def get_std_colors():
    d = {'White':(0,30,220),'Yellow':(30,160,200),'Orange':(12,200,240), 'Red':(0,210,180),'Green':(60,180,150),'Blue':(110,180,160)}
    for k, v in st.session_state.custom_std_colors.items(): d[k] = tuple(v)
    return d

def classify_color_lab(bgr_pixel, std_colors):
    pixel_mat = np.uint8([[bgr_pixel]])
    lab_pixel = cv2.cvtColor(pixel_mat, cv2.COLOR_BGR2LAB)[0][0]
    min_dist = float('inf'); best_color = 'White'
    for color_name, hsv_val in std_colors.items():
        std_bgr = cv2.cvtColor(np.uint8([[[hsv_val[0], hsv_val[1], hsv_val[2]]]]), cv2.COLOR_HSV2BGR)
        std_lab = cv2.cvtColor(std_bgr, cv2.COLOR_BGR2LAB)[0][0]
        dist = np.linalg.norm(lab_pixel.astype(float) - std_lab.astype(float))
        if dist < min_dist:
            min_dist = dist; best_color = color_name
    return best_color

def _grid_colors_with_pixels(warped, std_colors, classifier_fn):
    detected = ['White']*9; raw_bgrs = [np.zeros(3, dtype=np.uint8)]*9
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+.5)*100), int((r+.5)*100)
            roi = warped[max(0,ty-10):min(300,ty+10), max(0,tx-10):min(300,tx+10)]
            if roi.size > 0: bgr = np.median(roi, axis=(0,1)).astype(np.uint8)
            else: bgr = np.zeros(3, dtype=np.uint8)
            detected[r*3+c] = classifier_fn(bgr)
            raw_bgrs[r*3+c] = bgr
    return detected, raw_bgrs

def run_method_a(raw_bytes):
    # Method A: Standard OpenCV
    arr = np.frombuffer(raw_bytes, dtype=np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]; gs = int(min(h,w)*0.7); ox, oy = (w-gs)//2, (h-gs)//2
    warped = cv2.resize(img[oy:oy+gs, ox:ox+gs], (300, 300))
    std = get_std_colors()
    det, raw_bgrs = _grid_colors_with_pixels(warped, std, lambda b: classify_color_lab(b, std))
    return det, raw_bgrs, cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), None

def run_method_b(raw_bytes):
    # Method B: YOLOv8 Integration
    try:
        import yolo_detect
        # Ensure best.pt is in the same directory
        model_path = os.path.join(os.path.dirname(__file__), "best.pt")
        result = yolo_detect.get_cube_bbox(raw_bytes, model_path=model_path, draw=True)
        if result:
            cropped_300 = cv2.resize(result["cropped"], (300, 300))
            std = get_std_colors()
            det, raw_bgrs = _grid_colors_with_pixels(cropped_300, std, lambda b: classify_color_lab(b, std))
            return det, raw_bgrs, cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB), None
        return None, None, None, "❌ YOLO failed to detect a cube face."
    except ImportError:
        return None, None, None, "❌ yolo_detect.py not found."
    except Exception as e:
        return None, None, None, f"❌ YOLO Error: {str(e)}"

# ══════════════════════════════════════════════════════════════════════════════
# UI RENDERERS
# ══════════════════════════════════════════════════════════════════════════════
def render_live_cube_map(active_face):
    cube = st.session_state.cube_state; confirmed = st.session_state.confirmed_faces
    def face_html(f):
        is_act = (f == active_face); is_conf = (f in confirmed)
        t_col = "#6366f1" if is_act else ("#22c55e" if is_conf else "#94a3b8")
        cells = "".join([f'<div style="width:20px;height:20px;background:{HEX_COLORS[cube[f][i]]};border-radius:2px;"></div>' for i in range(9)])
        return f'<div style="display:flex;flex-direction:column;align-items:center;"><div style="font-size:9px;color:{t_col};font-weight:700;">{f}</div><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2px;border:1px solid #ddd;padding:2px;border-radius:4px;">{cells}</div></div>'
    
    html = f'''<div style="display:grid;grid-template-columns:repeat(4,70px);gap:10px;justify-content:center;width:max-content;margin:0 auto;font-family:sans-serif;">
        <div style="grid-column:2;">{face_html('Up')}</div>
        <div style="grid-row:2;">{face_html('Left')}</div><div style="grid-row:2;">{face_html('Front')}</div><div style="grid-row:2;">{face_html('Right')}</div><div style="grid-row:2;">{face_html('Back')}</div>
        <div style="grid-column:2;grid-row:3;">{face_html('Down')}</div>
    </div>'''
    components.html(html, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP FLOW
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧊 Console")
    # 🌟 修正點：確保 app_mode 在任何情況下都會被定義
    app_mode = st.radio("App Mode", ["🧩 Scan & Solve", "⚙️ Calibration"], label_visibility="collapsed")
    st.divider()
    if st.button("🗑️ Reset Cube", use_container_width=True):
        st.session_state.cube_state = {f:(['White']*4+[CENTER_COLORS[f]]+['White']*4) for f in FACES}
        st.session_state.confirmed_faces = []; st.session_state.last_solution = None
        st.rerun()

st.markdown('<div class="app-title">🧊 AI Rubik\'s Vision Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Multi-Algorithm Comparison & Topology Validation</div>', unsafe_allow_html=True)

if app_mode == "🧩 Scan & Solve":
    curr = st.session_state.active_face
    
    # Navigation
    nav_cols = st.columns(6)
    for i, f in enumerate(FACES):
        if nav_cols[i].button(f"{COLOR_EMOJIS[CENTER_COLORS[f]]} {f}", type="primary" if f==curr else "secondary", use_container_width=True):
            st.session_state.active_face = f; st.session_state.scan_result = None; st.rerun()

    col_l, col_r, col_map = st.columns([3, 2, 2], gap="large")

    with col_l:
        st.markdown("#### 📂 Photo Assist")
        up = st.file_uploader("Upload Face Photo", type=['jpg','png','jpeg'], label_visibility="collapsed")
        algo = st.selectbox("Vision Engine", ["OpenCV (Rule-based)", "YOLOv8 (Object Detection)"])
        
        if up:
            raw = up.getvalue()
            scan_key = f"scanned_{curr}_{algo}"
            if scan_key not in st.session_state or st.session_state[scan_key] != raw:
                with st.spinner("Analyzing..."):
                    det, raw_bgrs, overlay, err = run_method_a(raw) if "OpenCV" in algo else run_method_b(raw)
                    if not err:
                        det[4] = CENTER_COLORS[curr] # Force center
                        st.session_state.cube_state[curr] = det
                        st.session_state.scan_result = {"overlay": overlay, "face": curr}
                        st.session_state[scan_key] = raw
                        if curr not in st.session_state.confirmed_faces: st.session_state.confirmed_faces.append(curr)
                        push_history(); st.rerun()
                    else: st.error(err)
            
            if st.session_state.scan_result:
                st.image(st.session_state.scan_result["overlay"], caption=f"Analyzed via {algo}", use_container_width=True)

    with col_r:
        st.markdown('<span style="font-size:11px;font-weight:800;text-transform:uppercase;color:#64748b;">✏️ Manual Override</span>', unsafe_allow_html=True)
        C_SEQ = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
        def cycle_stk(f, ix):
            cur = st.session_state.cube_state[f][ix]
            st.session_state.cube_state[f][ix] = C_SEQ[(C_SEQ.index(cur)+1)%6]
            if f not in st.session_state.confirmed_faces: st.session_state.confirmed_faces.append(f)
            st.session_state.last_solution = None; push_history()

        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r*3+c; cv = st.session_state.cube_state[curr][idx]
                if idx==4: cols[c].button(f"🔒{COLOR_EMOJIS[cv]}", disabled=True, use_container_width=True)
                else: cols[c].button(f"{COLOR_EMOJIS[cv]}", key=f"g_{curr}_{idx}", on_click=cycle_stk, args=(curr, idx), use_container_width=True)

    with col_map:
        render_live_cube_map(curr)

    # Physics Check & Solve
    st.divider()
    all_colors = [s for f in FACES for s in st.session_state.cube_state[f]]
    counts = Counter(all_colors)
    errs = [c for c in HEX_COLORS if counts[c]!=9]
    
    if not errs:
        st.success("✨ Sticker count correct! Validating topology...")
        if st.button("🚀 Solve Cube", type="primary", use_container_width=True):
            success, res = solve_cube(st.session_state.cube_state)
            if success: st.session_state.last_solution = res; st.rerun()
            else: st.error(res)
    else:
        st.info("💡 Progress: " + " ".join([f"{COLOR_EMOJIS[c]} {counts[c]}/9" for c in counts]))

    if st.session_state.last_solution:
        st.success(f"✅ Solution: {st.session_state.last_solution}")

elif app_mode == "⚙️ Calibration":
    st.info("Calibration Mode: Upload a photo of a single color face to tune OpenCV thresholds.")
