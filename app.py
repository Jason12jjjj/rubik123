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
# CSS (MINIMALIST & SAFE AREA FOR TITLE)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    font-family:'Outfit',sans-serif!important;
    background: radial-gradient(circle at 0% 0%, #f8fafc 0%, #e2e8f0 100%)!important;
    color:#1e293b!important;
}

[data-testid="stHeader"] { background-color: transparent !important; }
[data-testid="stMainBlockContainer"]{ padding-top: 50px !important; }

.app-title { font-size: 2.8rem; font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; line-height: 1.2; }
.app-subtitle { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; font-weight: 500; letter-spacing: 0.5px; }

.mcard{
    background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 28px; padding: 32px; margin-bottom: 24px;
    box-shadow: 0 20px 40px -15px rgba(0,0,0,0.05), 0 5px 15px -5px rgba(0,0,0,0.02);
}
.slabel{ font-size: 11px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; color: #64748b; margin-bottom: 14px; display: block; opacity: 0.9; }

.stButton>button{
    border-radius: 12px!important; font-family: 'Outfit',sans-serif!important; font-weight: 800!important;
    background: #ffffff!important; border: 1.5px solid #f1f5f9!important;
    box-shadow: inset 0 -4px 6px rgba(0,0,0,0.03), 0 4px 10px -2px rgba(0,0,0,0.05)!important;
    transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)!important;
}
.stButton>button:hover{ transform: translateY(-2px) scale(1.03)!important; box-shadow: 0 12px 20px -5px rgba(0,0,0,0.1)!important; border-color: #6366f1!important; }
.stButton>button:active{ transform: scale(0.95)!important; }

.action-row{display:flex; gap:12px; margin-top:24px; padding-top:24px; border-top:1px solid rgba(0,0,0,0.03);}
.sol-box{
    background: rgba(248, 250, 252, 0.8); border-radius: 20px; padding: 24px;
    font-family: 'Courier New', monospace; font-size: 16px; font-weight: 800; box-shadow: inset 0 2px 8px rgba(0,0,0,0.04);
}
[data-testid="stSidebar"]{ background: rgba(255, 255, 255, 0.8)!important; backdrop-filter: blur(10px); border-right: 1px solid rgba(0,0,0,0.05)!important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
FACES         = ['Up','Left','Front','Right','Back','Down']
HEX_COLORS    = {'White':'#f1f5f9','Red':'#ef4444','Green':'#22c55e', 'Yellow':'#eab308','Orange':'#f97316','Blue':'#3b82f6'}
COLOR_EMOJIS  = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green', 'Right':'Red','Back':'Blue','Down':'Yellow'}
CALIB_FILE    = "calibration_profile.json"

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
    'scan_result':    None,
}

if 'custom_std_colors' not in st.session_state and os.path.exists(CALIB_FILE):
    try:
        with open(CALIB_FILE) as fh: _DEFAULTS['custom_std_colors'] = json.load(fh)
    except Exception: pass

for k, v in _DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

if st.session_state.history is None:
    st.session_state.history = [json.dumps({"cube_state": st.session_state.cube_state, "confirmed_faces": st.session_state.confirmed_faces})]

def push_history():
    sj = json.dumps({"cube_state": st.session_state.cube_state, "confirmed_faces": st.session_state.confirmed_faces})
    if st.session_state.history_index < len(st.session_state.history)-1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index+1]
    st.session_state.history.append(sj)
    st.session_state.history_index = len(st.session_state.history)-1

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS & OpenCV FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_std_colors():
    d = {'White':(0,30,220),'Yellow':(30,160,200),'Orange':(12,200,240), 'Red':(0,210,180),'Green':(60,180,150),'Blue':(110,180,160)}
    for k, v in st.session_state.custom_std_colors.items(): d[k] = tuple(v)
    return d

def face_complete(f): return f in st.session_state.get('confirmed_faces', [])

def mark_confirmed(face):
    cf = st.session_state.confirmed_faces
    if face not in cf: cf.append(face)

def unmark_confirmed(face):
    cf = st.session_state.confirmed_faces
    if face in cf: cf.remove(face)

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

def classify_color_mlp(bgr_pixel): return "White"
    
def _warp_to_300(img_bgr):
    h, w = img_bgr.shape[:2]
    gs = int(min(h,w)*0.7); ox, oy = (w-gs)//2, (h-gs)//2
    return cv2.resize(img_bgr[oy:oy+gs, ox:ox+gs], (300,300))

def _grid_colors_with_pixels(warped, std_colors, classifier_fn):
    detected = ['White']*9; raw_bgrs = [np.zeros(3, dtype=np.uint8)]*9
    hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV); sat_w = hsv_w[:,:,1]
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+.5)*100), int((r+.5)*100)
            y1,y2 = max(0,ty-35), min(300,ty+35); x1,x2 = max(0,tx-35), min(300,tx+35)
            moms = cv2.moments(sat_w[y1:y2,x1:x2]); fx, fy = tx, ty
            if moms["m00"] > 50:
                sl = x1+int(moms["m10"]/moms["m00"]); sm = y1+int(moms["m01"]/moms["m00"])
                if np.sqrt((sl-tx)**2+(sm-ty)**2) < 30: fx, fy = sl, sm
            roi = warped[max(0,fy-8):min(300,fy+8), max(0,fx-8):min(300,fx+8)]
            
            # 🛡️ 修復2: OpenCV 極端反光崩潰保護
            if roi.size > 0:
                rh, rw = roi.shape[:2]; c_ = roi[rh//4:rh-rh//4, rw//4:rw-rw//4]
                if c_.size > 0:
                    bgr = np.median(c_, axis=(0,1)).astype(np.uint8)
                else: bgr = np.zeros(3, dtype=np.uint8)
            else: bgr = np.zeros(3, dtype=np.uint8)
            
            detected[r*3+c] = classifier_fn(bgr)
            raw_bgrs[r*3+c] = bgr
    return detected, raw_bgrs

def _draw_grid_overlay(warped_rgb):
    vis = warped_rgb.copy(); h, w = vis.shape[:2]
    for i in range(1, 3):
        cv2.line(vis, (i*w//3, 0), (i*w//3, h), (100, 100, 255), 2)
        cv2.line(vis, (0, i*h//3), (w, i*h//3), (100, 100, 255), 2)
    cv2.rectangle(vis, (1,1), (w-2,h-2), (100, 100, 255), 3)
    return vis

def run_method_a(raw_bytes, expected_center):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None, None, None, "❌ Cannot decode image."
    std = get_std_colors(); warped = _warp_to_300(img)
    det, raw_bgrs = _grid_colors_with_pixels(warped, std, lambda b: classify_color_lab(b, std))
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return det, raw_bgrs, _draw_grid_overlay(warped_rgb), None

# ══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENTS (Live Map & 3D Player & Feedback) 
# ══════════════════════════════════════════════════════════════════════════════
def render_live_cube_map(active_face):
    cube = st.session_state.cube_state; confirmed = st.session_state.confirmed_faces
    def face_html(face_name):
        is_act = (face_name == active_face); is_conf = (face_name in confirmed)
        t_col = "#6366f1" if is_act else ("#22c55e" if is_conf else "#94a3b8")
        b_col = "#6366f1" if is_act else ("#22c55e" if is_conf else "transparent")
        shadow = "0 0 12px rgba(99,102,241,0.3)" if is_act else ("0 0 8px rgba(34,197,94,0.2)" if is_conf else "none")
        icon = "✏️" if is_act else ("✅" if is_conf else "⭕")
        cells = "".join([f'<div style="width:22px;height:22px;border-radius:3px;background:{HEX_COLORS.get(cube[face_name][i], "#f1f5f9")};"></div>' for i in range(9)])
        return f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;"><div style="font-size:10px;font-weight:700;text-align:center;color:{t_col};margin-bottom:4px;">{icon} {face_name}</div><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2px;border-radius:8px;border:2px solid {b_col};box-shadow:{shadow};padding:2px;">{cells}</div></div>'
    
    html = f'''<html><body style="margin:0;padding:0;background:transparent;font-family:Outfit,sans-serif;">
    <div style="background:rgba(255,255,255,0.9);border-radius:20px;padding:20px;border:1px solid rgba(0,0,0,0.06);box-shadow:0 8px 24px -8px rgba(0,0,0,0.06); overflow-x:auto;">
        <div style="font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#64748b;margin-bottom:16px;text-align:center;">🗺️ LIVE CUBE MAP</div>
        <div style="display:grid; grid-template-columns:78px 78px 78px 78px; gap:6px; justify-content:center; width:max-content; margin:0 auto;">
            <div style="grid-column:2;">{face_html('Up')}</div>
            <div style="grid-column:1; grid-row:2;">{face_html('Left')}</div>
            <div style="grid-column:2; grid-row:2;">{face_html('Front')}</div>
            <div style="grid-column:3; grid-row:2;">{face_html('Right')}</div>
            <div style="grid-column:4; grid-row:2;">{face_html('Back')}</div>
            <div style="grid-column:2; grid-row:3;">{face_html('Down')}</div>
        </div>
        <div style="text-align:center;margin-top:16px;font-size:10px;color:#94a3b8;font-weight:700;">✅ {len(confirmed)}/6 FACES CONFIRMED</div>
    </div></body></html>'''
    components.html(html, height=400)

def render_detection_feedback(scan_result):
    if not scan_result: return
    det_colors, raw_bgrs, engine, face = scan_result.get('detected', []), scan_result.get('raw_bgrs', []), scan_result.get('engine', 'OpenCV'), scan_result.get('face', 'Front')
    st.markdown(f"##### 🔍 Detection Result — {face}")
    if scan_result.get('overlay') is not None: st.image(scan_result['overlay'], caption=f"📐 How {engine} analyzed your photo", use_container_width=True)
    st.markdown("---")
    
    c_html = "".join([f'<div style="width:52px;height:52px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;border:2px solid rgba(255,255,255,0.6);background:{HEX_COLORS.get(c,"#f1f5f9")};color:{"#333" if c in ["White","Yellow"] else "white"};">{c[:3]}</div>' for c in det_colors])
    p_html = "".join([f'<div style="width:52px;height:52px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:8px;border:2px solid rgba(255,255,255,0.6);background:rgb({b[2]},{b[1]},{b[0]});color:white;">R{int(b[2])}<br>G{int(b[1])}<br>B{int(b[0])}</div>' for b in raw_bgrs])
    
    c1, c2 = st.columns(2)
    with c1: st.markdown("**AI Classification:**"); st.markdown(f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;max-width:180px;">{c_html}</div>', unsafe_allow_html=True)
    with c2: st.markdown("**Raw Pixel Colors:**"); st.markdown(f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;max-width:180px;">{p_html}</div>', unsafe_allow_html=True)

def render_3d_player(solution):
    def inv(s):
        r=[]
        for m in reversed(s.split()):
            if "'" in m: r.append(m.replace("'",""))
            elif "2" in m: r.append(m)
            else: r.append(m+"'")
        return " ".join(r)
    speed = st.session_state.get('solve_speed',1.0)
    html = f"""<div style="background:#f8fafc; border-radius:18px; padding:14px; border:1px solid #e2e8f0;"><script type="module" src="https://cdn.cubing.net/js/cubing/twisty"></script><twisty-player experimental-setup-alg="{inv(solution)}" alg="{solution}" background="none" tempo-scale="{speed}" control-panel="bottom-row" style="width:100%; height:380px;"></twisty-player></div>"""
    components.html(html, height=430)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2 style='margin-top:0;'>🧊 Console</h2>", unsafe_allow_html=True)
    app_mode = st.radio("Mode", ["🧩 Scan & Solve", "⚙️ Calibration"], label_visibility="collapsed")
    st.divider()
    if app_mode == "🧩 Scan & Solve":
        with st.expander("📊 Sticker Stats", expanded=True):
            all_s = [s for f in FACES for s in st.session_state.cube_state[f]]
            for name in HEX_COLORS:
                cnt = all_s.count(name); ok = (cnt==9)
                st.markdown(f"<div style='font-size:12px; font-weight:600;'>{COLOR_EMOJIS[name]} {name}: <span style='color:{'#22c55e' if ok else '#ef4444'};'>{cnt}/9</span></div>", unsafe_allow_html=True)
        if st.button("🗑️ Reset Cube", use_container_width=True):
            st.session_state.cube_state = {f:(['White']*4+[CENTER_COLORS[f]]+['White']*4)for f in FACES}
            st.session_state.confirmed_faces = []; st.session_state.last_solution = None; st.session_state.scan_result = None
            push_history(); st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

# 🛡️ 修復3: 全局顯示標題，無論在哪個模式都不會消失
st.markdown('''
    <div class="app-title">🧊 AI Rubik's Vision Engine</div>
    <div class="app-subtitle">Multi-Algorithm Comparison & Topology Validation System</div>
''', unsafe_allow_html=True)

if app_mode == "🧩 Scan & Solve":
    curr = st.session_state.active_face
    
    # ── One-Line Navigation & Palette ───────────────────────────────────────
    pw_cols = st.columns(6)
    for i, f in enumerate(FACES):
        cc = CENTER_COLORS[f]
        is_act = (f == curr); is_conf = face_complete(f)
        lbl = f"✅ {f}" if (is_conf and not is_act) else f"{COLOR_EMOJIS[cc]} {f}"
        if pw_cols[i].button(lbl, key=f"pwr_{f}", use_container_width=True, type="primary" if is_act else "secondary"):
            st.session_state.active_face = f; st.session_state.selected_color = cc; st.session_state.scan_result = None
            st.rerun()

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # ── Main 3-Column Layout: Upload | Grid | Live Map ──────────────────────
    col_l, col_r, col_map = st.columns([3, 2, 2], gap="large")
    
    with col_l:
        st.markdown("#### 📂 Photo Assist")
        up = st.file_uploader("Upload reference", type=['jpg','png','jpeg'], key=f"up_{curr}", label_visibility="collapsed")
        
        if up:
            raw = up.getvalue() 
            st.image(raw, caption="📷 Your uploaded photo", use_container_width=True)
            st.divider()
            
            st.markdown("##### 🔬 Vision Engine")
            algo_choice = st.selectbox(
                "Select AI Model:",
                ["📐 OpenCV (Math Distance)", "🎯 YOLOv8 (Object Detection)", "🧠 SVM (Machine Learning)"],
                label_visibility="collapsed", key=f"algo_sel_{curr}"
            )

            engine_name = algo_choice.split(" ")[1]
            scan_key = f"scanned_{curr}_{algo_choice}"
            
            # AUTO-SCAN WITH FULL ERROR DEFENSE
            if scan_key not in st.session_state or st.session_state[scan_key] != raw:
                st.session_state[scan_key] = raw 
                
                with st.spinner(f"Auto-Analyzing via {engine_name}..."):
                    det, raw_bgrs, overlay, err = None, None, None, None
                    
                    try:
                        if "OpenCV" in algo_choice:
                            det, raw_bgrs, overlay, err = run_method_a(raw, CENTER_COLORS[curr])
                            
                        elif "YOLOv8" in algo_choice:
                            try:
                                import yolo_detect
                                err = "🚧 YOLO Module Pending Integration from Team Member A."
                            except ImportError:
                                err = "❌ Missing File: Cannot find 'yolo_detect.py'."
                            except Exception as e:
                                err = f"❌ YOLO Model Error: {str(e)}"
                                
                        elif "SVM" in algo_choice:
                            try:
                                import svm_predict
                                err = "🚧 SVM Module Pending Integration from Team Member B."
                            except ImportError:
                                err = "❌ Missing File: Cannot find 'svm_predict.py'."
                            except Exception as e:
                                err = f"❌ SVM Model Error: {str(e)}"
                                
                    except Exception as fatal_e:
                        err = f"🚨 Fatal Vision Engine Error: {str(fatal_e)}"

                    if err:
                        st.error(err)
                    elif det:
                        det[4] = CENTER_COLORS[curr]
                        st.session_state.cube_state[curr] = det
                        # 🛡️ 修復1: 確保每次 AI 改變魔方狀態時，清空舊的解答
                        st.session_state.last_solution = None 
                        mark_confirmed(curr); push_history()
                        st.session_state.scan_result = {
                            'detected': det,
                            'raw_bgrs': [bgr.tolist() if hasattr(bgr, 'tolist') else list(bgr) for bgr in raw_bgrs],
                            'overlay': overlay, 'engine': engine_name, 'face': curr,
                        }
                        st.rerun()
            
            # Persistent Feedback Panel
            if st.session_state.scan_result and st.session_state.scan_result.get('face') == curr:
                sr = st.session_state.scan_result
                render_detection_feedback({
                    'detected': sr['detected'],
                    'raw_bgrs': [np.array(b, dtype=np.uint8) for b in sr['raw_bgrs']],
                    'overlay': sr.get('overlay'), 'engine': sr['engine'], 'face': sr['face'],
                })
                st.markdown("")
                bc1, bc2 = st.columns(2)
                if bc1.button("✅ Accept & Next", type="primary", use_container_width=True):
                    st.session_state.scan_result = None
                    next_idx = (FACES.index(curr)+1) % 6
                    remaining = [f for f in FACES if not face_complete(f)]
                    st.session_state.active_face = remaining[0] if remaining else FACES[next_idx]
                    st.rerun()
                if bc2.button("🔄 Retry Scan", use_container_width=True):
                    unmark_confirmed(curr)
                    st.session_state.cube_state[curr] = ['White']*4+[CENTER_COLORS[curr]]+['White']*4
                    st.session_state.scan_result = None
                    st.session_state.last_solution = None # 🛡️ 修復1: 清空舊解答
                    if scan_key in st.session_state: del st.session_state[scan_key]
                    push_history(); st.rerun()
        else:
            st.info(f"📷 Upload a photo of the **{curr}** face (center = {COLOR_EMOJIS[CENTER_COLORS[curr]]} {CENTER_COLORS[curr]})")
    
    with col_r:
        st.markdown('<span class="slabel">✏️ Manual Override</span>', unsafe_allow_html=True)
        C_SEQ = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
        def cycle_stk(face, ix):
            st.session_state.cube_state[face][ix] = C_SEQ[(C_SEQ.index(st.session_state.cube_state[face][ix]) + 1) % len(C_SEQ)]
            st.session_state.last_solution = None # 🛡️ 修復1: 確保手動改顏色時，清空舊解答
            mark_confirmed(face); push_history()

        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r*3+c; cv = st.session_state.cube_state[curr][idx]
                if idx==4: cols[c].button(f"🔒{COLOR_EMOJIS[cv]}", disabled=True, use_container_width=True)
                else: cols[c].button(f"{COLOR_EMOJIS[cv]}", key=f"g_{curr}_{idx}", on_click=cycle_stk, args=(curr, idx), use_container_width=True)
    
    with col_map:
        render_live_cube_map(curr)

    # ── Result & Physics Validation Section ──────────────────────────────────
    st.divider()
    all_s = [s for f in FACES for s in st.session_state.cube_state[f]]
    errs = [c for c in HEX_COLORS if all_s.count(c)!=9]
    
    if not errs:
        st.success("✨ Sticker count is correct! Ready to validate physics.")
        if st.button("⚡ Solve Cube (Physics Check)", use_container_width=True, type="primary"):
            is_success, result_message = solve_cube(st.session_state.cube_state)
            if not is_success:
                st.error(result_message) 
            else:
                st.session_state.last_solution = result_message
                st.rerun()
                
    elif st.session_state.last_solution is None:
        st.warning("⚠️ Cannot solve yet. Please fix sticker counts:")
        st.info("💡 Progress: " + ", ".join([f"{COLOR_EMOJIS[c]} {all_s.count(c)}/9" for c in errs]))

    if st.session_state.last_solution:
        st.markdown('<div class="mcard">', unsafe_allow_html=True)
        st.markdown(f'<div class="sol-box">✅ Route: {st.session_state.last_solution}</div>', unsafe_allow_html=True)
        render_3d_player(st.session_state.last_solution)
        st.markdown('</div>', unsafe_allow_html=True)

if app_mode == "⚙️ Calibration":
    st.markdown('<div class="mcard"><h2>⚙️ Environment Calibration</h2><p>Adjust OpenCV HSV/LAB thresholds here. (Module in development)</p></div>', unsafe_allow_html=True)
