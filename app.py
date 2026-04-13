import os, json
from collections import Counter
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from rubiks_core import (
    validate_cube_state, solve_cube,
    classify_color_lab, classify_color_hsv, classify_color_knn, classify_color_mlp,
    extract_center_bgr, COLORS,
)
from yolo_detect import get_cube_bbox, get_face_colors_from_crop, detect_stickers

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Rubik's AI Solver", page_icon="🧊",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# CSS (MINIMALIST)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── Global Studio Aesthetic ── */
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    font-family:'Outfit',sans-serif!important;
    background: radial-gradient(circle at 0% 0%, #f8fafc 0%, #e2e8f0 100%)!important;
    color:#1e293b!important;
}
[data-testid="stMainBlockContainer"]{padding-top:40px!important;}

/* ── Glassmorphic Cards ── */
.mcard{
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 28px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.05), 0 5px 15px -5px rgba(0,0,0,0.02);
}
.slabel{
    font-size: 11px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase;
    color: #64748b; margin-bottom: 14px; display: block; opacity: 0.9;
}

/* ── Premium Control Bar ── */
.power-btn{
    border-radius: 14px!important; font-weight: 700!important; 
    border: 1px solid rgba(255,255,255,0.8)!important;
    background: rgba(255,255,255,0.5)!important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05)!important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1)!important;
}

/* ── Tactile Grid Stickers ── */
.stButton>button{
    border-radius: 12px!important; 
    font-family: 'Outfit',sans-serif!important;
    font-weight: 800!important;
    background: #ffffff!important;
    border: 1.5px solid #f1f5f9!important;
    box-shadow: inset 0 -4px 6px rgba(0,0,0,0.03), 0 4px 10px -2px rgba(0,0,0,0.05)!important;
    transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)!important;
}
.stButton>button:hover{
    transform: translateY(-2px) scale(1.03)!important;
    box-shadow: 0 12px 20px -5px rgba(0,0,0,0.1)!important;
    border-color: #6366f1!important;
}
.stButton>button:active{transform: scale(0.95)!important;}

/* ── Action Footer ── */
.action-row{display:flex; gap:12px; margin-top:24px; padding-top:24px; border-top:1px solid rgba(0,0,0,0.03);}

/* ── Solution & Sidebar ── */
.sol-box{
    background: rgba(248, 250, 252, 0.8);
    border-radius: 20px; padding: 24px;
    font-family: 'Courier New', monospace; font-size: 16px; font-weight: 800;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.04);
}

[data-testid="stSidebar"]{
    background: rgba(255, 255, 255, 0.8)!important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(0,0,0,0.05)!important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FACES         = ['Up','Left','Front','Right','Back','Down']
HEX_COLORS    = {'White':'#f1f5f9','Red':'#ef4444','Green':'#22c55e',
                  'Yellow':'#eab308','Orange':'#f97316','Blue':'#3b82f6'}
COLOR_EMOJIS  = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green',
                  'Right':'Red','Back':'Blue','Down':'Yellow'}
CALIB_FILE    = "calibration_profile.json"

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    'active_face':    'Front',
    'cube_state':     {f: (['White']*4+[CENTER_COLORS[f]]+['White']*4) for f in FACES},
    'last_solution':  None,
    'selected_color': 'White',
    'solve_speed':    1.0,
    'custom_std_colors': {},
    'history':        None,
    'history_index':  0,
    'confirmed_faces': [],
}

if 'custom_std_colors' not in st.session_state and os.path.exists(CALIB_FILE):
    try:
        with open(CALIB_FILE) as fh:
            _DEFAULTS['custom_std_colors'] = json.load(fh)
    except Exception: pass

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.history is None:
    st.session_state.history = [json.dumps({
        "cube_state": st.session_state.cube_state,
        "confirmed_faces": st.session_state.confirmed_faces
    })]

def push_history():
    sj = json.dumps({
        "cube_state": st.session_state.cube_state,
        "confirmed_faces": st.session_state.confirmed_faces
    })
    if st.session_state.history_index < len(st.session_state.history)-1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index+1]
    st.session_state.history.append(sj)
    st.session_state.history_index = len(st.session_state.history)-1

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_std_colors():
    d = {'White':(0,30,220),'Yellow':(30,160,200),'Orange':(12,200,240),
         'Red':(0,210,180),'Green':(60,180,150),'Blue':(110,180,160)}
    for k, v in st.session_state.custom_std_colors.items(): d[k] = tuple(v)
    return d

def hex_to_bgr(h):
    h = h.lstrip('#')
    return (int(h[4:6],16), int(h[2:4],16), int(h[0:2],16))

def face_complete(f): return f in st.session_state.get('confirmed_faces', [])

def mark_confirmed(face):
    cf = st.session_state.confirmed_faces
    if face not in cf: cf.append(face)

def unmark_confirmed(face):
    cf = st.session_state.confirmed_faces
    if face in cf: cf.remove(face)

# ══════════════════════════════════════════════════════════════════════════════
# DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def _warp_to_300(img_bgr):
    h, w = img_bgr.shape[:2]
    gs = int(min(h,w)*0.7); ox, oy = (w-gs)//2, (h-gs)//2
    return cv2.resize(img_bgr[oy:oy+gs, ox:ox+gs], (300,300))

def _grid_colors(warped, std_colors, classifier_fn):
    detected = ['White']*9
    hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV); sat_w = hsv_w[:,:,1]
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+.5)*100), int((r+.5)*100)
            y1,y2 = max(0,ty-35), min(300,ty+35); x1,x2 = max(0,tx-35), min(300,tx+35)
            moms = cv2.moments(sat_w[y1:y2,x1:x2])
            fx, fy = tx, ty
            if moms["m00"] > 50:
                sl = x1+int(moms["m10"]/moms["m00"]); sm = y1+int(moms["m01"]/moms["m00"])
                if np.sqrt((sl-tx)**2+(sm-ty)**2) < 30: fx, fy = sl, sm
            roi = warped[max(0,fy-8):min(300,fy+8), max(0,fx-8):min(300,fx+8)]
            if roi.size > 0:
                rh, rw = roi.shape[:2]; c_ = roi[rh//4:rh-rh//4, rw//4:rw-rw//4]
                bgr = np.median(c_, axis=(0,1)).astype(np.uint8)
            else: bgr = np.zeros(3, dtype=np.uint8)
            detected[r*3+c] = classifier_fn(bgr)
    return detected

def run_method_a(raw_bytes, expected_center):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None, None, "❌ Cannot decode image."
    std = get_std_colors(); warped = _warp_to_300(img)
    det = _grid_colors(warped, std, lambda b: classify_color_lab(b, std))
    return det, cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), None

def run_method_b(raw_bytes, expected_center):
    """Method B: Real YOLOv8 Object Detection.
    
    1. Use get_cube_bbox() to locate the Rubik's cube in the photo.
    2. Crop the detected region.
    3. Divide cropped region into 3×3 grid and classify each sticker.
    Falls back to OpenCV method if no YOLO model is available.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, "❌ Cannot decode image."

    try:
        # Step 1 — Detect cube bounding box with YOLOv8
        result = get_cube_bbox(raw_bytes, draw=True)

        if result is None:
            # No cube detected — fall back to grid approach
            std = get_std_colors()
            warped = _warp_to_300(img)
            det = _grid_colors(warped, std, lambda b: classify_color_lab(b, std))
            return det, cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), "⚠️ YOLO: No cube detected – fell back to OpenCV"

        x1, y1, x2, y2 = result["bbox"]
        conf = result["confidence"]
        cropped = result["cropped"]

        # Step 2 — Resize cropped region to 300×300 for grid analysis
        warped = cv2.resize(cropped, (300, 300))

        # Step 3 — Classify 9 sticker colours from the cropped face
        std = get_std_colors()
        det = get_face_colors_from_crop(
            warped,
            classifier_fn=lambda b: classify_color_lab(b, std)
        )

        # Build preview image with YOLO bbox overlay
        preview = result["annotated"] if result["annotated"] is not None else img.copy()
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        return det, preview_rgb, f"✅ YOLOv8 detected cube (conf={conf:.2f})"

    except FileNotFoundError:
        # Model not found — graceful fallback
        std = get_std_colors()
        warped = _warp_to_300(img)
        det = _grid_colors(warped, std, lambda b: classify_color_lab(b, std))
        return det, cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), "⚠️ YOLO model (best.pt) not found – using OpenCV fallback"

def run_method_c(raw_bytes, expected_center):
    # Method C: SVM/MLP Classifier
    arr = np.frombuffer(raw_bytes, dtype=np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None, None, "❌ Cannot decode image."
    std = get_std_colors(); warped = _warp_to_300(img)
    det = _grid_colors(warped, std, lambda b: classify_color_mlp(b))
    return det, cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), "SVM/MLP Neural Network Active"

# ══════════════════════════════════════════════════════════════════════════════
# 3D PLAYER
# ══════════════════════════════════════════════════════════════════════════════
def render_3d_player(solution):
    def inv(s):
        r=[]
        for m in reversed(s.split()):
            if "'" in m: r.append(m.replace("'",""))
            elif "2" in m: r.append(m)
            else: r.append(m+"'")
        return " ".join(r)
    speed = st.session_state.get('solve_speed',1.0)
    html = f"""
    <div style="background:#f8fafc; border-radius:18px; padding:14px; border:1px solid #e2e8f0;">
      <script type="module" src="https://cdn.cubing.net/js/cubing/twisty"></script>
      <twisty-player experimental-setup-alg="{inv(solution)}" alg="{solution}"
        background="none" tempo-scale="{speed}" control-panel="bottom-row"
        style="width:100%; height:380px;"></twisty-player>
    </div>"""
    components.html(html, height=430)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2 style='margin-top:0;'>🧊 Solver</h2>", unsafe_allow_html=True)
    app_mode = st.radio("Mode", ["🧩 Scan & Solve", "⚙️ Calibration"], label_visibility="collapsed")
    st.divider()

    # ── YOLOv8 Model Status ──
    _yolo_model_path = os.environ.get(
        "YOLO_MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "runs", "detect", "rubik_cube", "weights", "best.pt"),
    )
    _yolo_available = os.path.isfile(_yolo_model_path)
    with st.expander("🎯 YOLOv8 Status", expanded=False):
        if _yolo_available:
            st.success(f"✅ Model loaded")
            st.caption(f"`{_yolo_model_path}`")
        else:
            st.warning("⚠️ No model found")
            st.caption("Run `train_yolo.py` to train, or place `best.pt` in the expected path.")
            st.code(f"{_yolo_model_path}", language="text")

    if app_mode == "🧩 Scan & Solve":
        with st.expander("📊 Sticker Stats"):
            all_s = [s for f in FACES for s in st.session_state.cube_state[f]]
            for name in HEX_COLORS:
                cnt = all_s.count(name); ok = (cnt==9)
                st.markdown(f"<div style='font-size:11px;'>{COLOR_EMOJIS[name]} {name}: <b>{cnt}/9</b></div>", unsafe_allow_html=True)
        if st.button("🗑️ Reset Cube", use_container_width=True):
            st.session_state.cube_state = {f:(['White']*4+[CENTER_COLORS[f]]+['White']*4)for f in FACES}
            st.session_state.confirmed_faces = []; st.session_state.last_solution = None
            push_history(); st.rerun()

    # ── MOBILE-FRIENDLY SHOOTING GUIDE (RESPONSIVE) ──
    st.divider()
    st.subheader("📱 Standard Cube Setup (Mobile Friendly)")

    st.markdown("""
    <style>
    .desktop-guide { display: block; }
    .mobile-guide { display: none; }
    @media (max-width: 768px) {
        .desktop-guide { display: none; }
        .mobile-guide { display: block; }
    }
    </style>

    <div class="desktop-guide">
    <p><b>Desktop Users:</b><br>
    Hold cube with <b>White center on Top</b> and <b>Green center facing you</b>.<br>
    Camera should be parallel to the face being scanned.</p>
    </div>

    <div class="mobile-guide">
    <p><b>Mobile Users (Important!):</b><br>
    1. <b>U (White)</b>: Place cube on table, phone directly above.<br>
    2. <b>F (Green)</b>: Keep White on top, hold cube parallel to phone screen.<br>
    3. <b>R/B/L</b>: Keep White pointing to ceiling, rotate cube horizontally.<br>
    4. <b>D (Yellow)</b>: Flip cube upside down, Yellow toward phone.<br>
    ⚠️ <b>Never change White's direction (always ceiling) during side scans.</b></p>
    </div>
    """, unsafe_allow_html=True)

# ── Title is now moved to sidebar for a cleaner studio layout ──

# ══════════════════════════════════════════════════════════════════════════════
# SCAN & SOLVE PAGE
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "🧩 Scan & Solve":
    curr = st.session_state.active_face
    # ── One-Line Navigation & Palette ───────────────────────────────────────
    pw_cols = st.columns(6)
    for i, f in enumerate(FACES):
        cc = CENTER_COLORS[f]
        is_act = (f == curr)
        lbl = f"{COLOR_EMOJIS[cc]} {f}" 
        btn_type = "primary" if is_act else "secondary"
        if pw_cols[i].button(lbl, key=f"pwr_{f}", use_container_width=True, type=btn_type):
            st.session_state.active_face = f
            st.session_state.selected_color = cc
            st.rerun()

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # Input Body
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("#### 📂 Photo Assist")
        up = st.file_uploader("Upload reference", type=['jpg','png','jpeg'], key=f"up_{curr}", label_visibility="collapsed")
        
        if up:
            raw = up.read()
            # 1. Show the uploaded original photo
            st.image(raw, use_container_width=True)
            
            st.divider()
            
            # 2. Vision Engine selector (clean dropdown)
            st.markdown("##### 🔬 Vision Engine")
            algo_choice = st.selectbox(
                "Select AI Model:",
                ["📐 OpenCV (Math Distance)", "🎯 YOLOv8 (Object Detection)", "🧠 SVM (Machine Learning)"],
                label_visibility="collapsed",
                key=f"algo_sel_{curr}"
            )

            # 3. Dynamic button label follows selected engine
            engine_name = algo_choice.split(" ")[1]  # Extract OpenCV, YOLOv8, or SVM
            if st.button(f"📸 Scan with {engine_name}", type="primary", use_container_width=True):
                
                with st.spinner(f"Analyzing via {engine_name}..."):
                    
                    det, img, err = None, None, None
                    
                    # 4. Algorithm routing
                    if "OpenCV" in algo_choice:
                        det, img, err = run_method_a(raw, CENTER_COLORS[curr])
                        
                    elif "YOLOv8" in algo_choice:
                        det, img, err = run_method_b(raw, CENTER_COLORS[curr])
                        
                    elif "SVM" in algo_choice:
                        det, img, err = run_method_c(raw, CENTER_COLORS[curr])

                    # 5. Unified result handling
                    if err and not isinstance(err, str): st.error("Detection Error")
                    elif isinstance(err, str) and "Active" in err: st.toast(err)

                    if det:
                        det[4] = CENTER_COLORS[curr]
                        st.session_state.cube_state[curr] = det
                        mark_confirmed(curr); push_history()
                        
                        # Debug View — shows how each engine "sees" the face
                        if img is not None:
                            with st.expander(f"👁️ {engine_name} Debug View", expanded=True):
                                st.image(img, use_container_width=True, caption=f"How {engine_name} sees it")
                        
                        # Brief pause so user can see results, then auto-advance
                        import time; time.sleep(1)
                        st.session_state.active_face = FACES[(FACES.index(curr)+1)%6]
                        st.rerun()
                    elif err: st.error(err)
    with col_r:
        st.markdown('<span class="slabel">✏️ Manual Grid</span>', unsafe_allow_html=True)
        
        # Color progression for cycling
        C_SEQ = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
        
        def cycle_stk(face, ix):
            cur_c = st.session_state.cube_state[face][ix]
            # Find next color in sequence
            next_c = C_SEQ[(C_SEQ.index(cur_c) + 1) % len(C_SEQ)]
            st.session_state.cube_state[face][ix] = next_c
            mark_confirmed(face); push_history()

        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r*3+c; cv = st.session_state.cube_state[curr][idx]
                if idx==4: cols[c].button(f"🔒{COLOR_EMOJIS[cv]}", disabled=True, use_container_width=True)
                else: cols[c].button(f"{COLOR_EMOJIS[cv]}", key=f"g_{curr}_{idx}", on_click=cycle_stk, args=(curr, idx), use_container_width=True)

    # Action Footer
    st.markdown('<div class="action-row">', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    if a1.button("🧹 Reset", use_container_width=True):
        st.session_state.cube_state[curr] = ['White']*4+[CENTER_COLORS[curr]]+['White']*4
        unmark_confirmed(curr); push_history(); st.rerun()
    if a2.button("🎨 Fill", use_container_width=True):
        st.session_state.cube_state[curr] = [sel]*4+[CENTER_COLORS[curr]]+[sel]*4
        mark_confirmed(curr); push_history(); st.rerun()
    if a3.button("🚀 Confirm", use_container_width=True, type="primary"):
        mark_confirmed(curr); rem = [f for f in FACES if not face_complete(f)]
        if rem: st.session_state.active_face = rem[0]
        st.rerun()
    st.markdown('', unsafe_allow_html=True)

    # Result Section
    all_s = [s for f in FACES for s in st.session_state.cube_state[f]]
    errs = [c for c in HEX_COLORS if all_s.count(c)!=9]
    if not errs:
        st.success("✨ Ready to solve!"); 
        if st.button("⚡ Solve Cube", use_container_width=True, type="primary"):
            sol, e = solve_cube(st.session_state.cube_state)
            if e: st.error(e)
            else: st.session_state.last_solution = sol; st.rerun()
    elif st.session_state.last_solution is None:
        st.info("💡 Progress: " + ", ".join([f"{COLOR_EMOJIS[c]} {all_s.count(c)}/9" for c in errs]))

    if st.session_state.last_solution:
        st.markdown('<div class="mcard">', unsafe_allow_html=True)
        st.markdown(f'<div class="sol-box">{st.session_state.last_solution}</div>', unsafe_allow_html=True)
        render_3d_player(st.session_state.last_solution)
        st.markdown('</div>', unsafe_allow_html=True)

if app_mode == "⚙️ Calibration":
    st.markdown('<div class="mcard">Settings & Calibration logic here...</div>', unsafe_allow_html=True)
