import os, json, io, base64
from collections import Counter
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from rubiks_core import (
    validate_cube_state, solve_cube,
    classify_color_lab, classify_color_hsv, classify_color_knn, classify_color_mlp,
    classify_color_svm,
    extract_center_bgr, COLORS,
)
import yolo_detect

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

/* ── Live Cube Map ── */
.cube-map-container {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 8px 24px -8px rgba(0,0,0,0.06);
}
.cube-map-title {
    font-size: 11px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase;
    color: #64748b; margin-bottom: 16px; text-align: center;
}
.face-mini {
    display: inline-block; margin: 3px; vertical-align: top;
}
.face-mini-title {
    font-size: 10px; font-weight: 700; text-align: center; color: #94a3b8;
    margin-bottom: 4px; letter-spacing: 1px;
}
.face-mini-title.active { color: #6366f1; font-weight: 800; }
.face-mini-title.confirmed { color: #22c55e; }
.mini-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 2px;
    border-radius: 8px; overflow: hidden; border: 2px solid transparent;
    transition: border-color 0.3s;
}
.mini-grid.active-grid { border-color: #6366f1; box-shadow: 0 0 12px rgba(99,102,241,0.25); }
.mini-grid.confirmed-grid { border-color: #22c55e; }
.mini-cell {
    width: 22px; height: 22px; border-radius: 3px;
    transition: transform 0.2s;
}

/* ── Detection Feedback ── */
.detection-card {
    background: rgba(255,255,255,0.9);
    border-radius: 16px;
    padding: 16px;
    border: 1px solid rgba(0,0,0,0.06);
    margin-top: 12px;
}
.det-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px;
    max-width: 180px; margin: 0 auto;
}
.det-cell {
    width: 52px; height: 52px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 800; color: rgba(0,0,0,0.5);
    border: 2px solid rgba(255,255,255,0.6);
    box-shadow: inset 0 -2px 4px rgba(0,0,0,0.1);
}
.pixel-strip {
    display: flex; gap: 2px; justify-content: center; margin-top: 8px;
}
.px-swatch {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid rgba(0,0,0,0.1);
}
.app-title {
    font-size: 2.8rem; font-weight: 800; text-align: center;
    background: linear-gradient(90deg, #1e293b, #474ef1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.app-subtitle {
    font-size: 0.9rem; font-weight: 600; text-align: center;
    color: #64748b; letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 40px;
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
    'scan_result':    None,   # stores last scan feedback
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
# DETECTION  (returns detected colors + raw BGR pixels + annotated image)
# ══════════════════════════════════════════════════════════════════════════════
def _warp_to_300(img_bgr):
    h, w = img_bgr.shape[:2]
    gs = int(min(h,w)*0.7); ox, oy = (w-gs)//2, (h-gs)//2
    return cv2.resize(img_bgr[oy:oy+gs, ox:ox+gs], (300,300))

def _grid_colors_with_pixels(warped, std_colors, classifier_fn, use_blocks=False):
    """Returns (detected_colors[9], raw_bgr_pixels[9])"""
    detected = ['White']*9
    raw_bgrs = [np.zeros(3, dtype=np.uint8)]*9
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
            
            # Extract ROI for median and block
            roi_size = 8 if not use_blocks else 25 # Larger for SVM features
            roi = warped[max(0,fy-roi_size):min(300,fy+roi_size), max(0,fx-roi_size):min(300,fx+roi_size)]
            
            if roi.size > 0:
                rh, rw = roi.shape[:2]; c_ = roi[rh//4:rh-rh//4, rw//4:rw-rw//4]
                median_bgr = np.median(c_, axis=(0,1)).astype(np.uint8)
                
                if use_blocks:
                    detected[r*3+c] = classifier_fn(roi) # SVM uses the whole ROI
                else:
                    detected[r*3+c] = classifier_fn(median_bgr)
                raw_bgrs[r*3+c] = median_bgr
            else:
                raw_bgrs[r*3+c] = np.zeros(3, dtype=np.uint8)
                detected[r*3+c] = "White"
                
    return detected, raw_bgrs

def _draw_grid_overlay(warped_rgb):
    """Draw a 3x3 grid overlay on the warped image for visual feedback."""
    vis = warped_rgb.copy()
    h, w = vis.shape[:2]
    # Grid lines
    for i in range(1, 3):
        cv2.line(vis, (i*w//3, 0), (i*w//3, h), (100, 100, 255), 2)
        cv2.line(vis, (0, i*h//3), (w, i*h//3), (100, 100, 255), 2)
    # Border
    cv2.rectangle(vis, (1,1), (w-2,h-2), (100, 100, 255), 3)
    # Center dots
    for r in range(3):
        for c in range(3):
            cx = int((c+0.5)*w/3)
            cy = int((r+0.5)*h/3)
            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
            cv2.circle(vis, (cx, cy), 6, (100, 100, 255), 2)
    return vis

def run_method_a(raw_bytes, expected_center):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None, None, None, "❌ Cannot decode image."
    std = get_std_colors(); warped = _warp_to_300(img)
    det, raw_bgrs = _grid_colors_with_pixels(warped, std, lambda b: classify_color_lab(b, std))
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    overlay = _draw_grid_overlay(warped_rgb)
    return det, raw_bgrs, overlay, None

def run_method_b(raw_bytes, expected_center):
    """YOLOv8 Detection Method"""
    try:
        # Get stickers AND annotated overlay
        overlay, stickers = yolo_detect.detect_and_draw(raw_bytes)
        
        if len(stickers) != 9:
            # HYBRID FALLBACK: If stickers are missing but cube is found
            cube_res = yolo_detect.get_cube_bbox(raw_bytes, draw=True)
            if cube_res:
                # 1. Use mathematical grid division
                det = yolo_detect.get_face_colors_from_crop(cube_res["cropped"], classifier_fn=lambda b: classify_color_lab(b, get_std_colors()))
                
                # 2. Extract raw BGRs for feedback
                h, w = cube_res["cropped"].shape[:2]
                ch, cw = h//3, w//3
                raw_bgrs = []
                for r in range(3):
                    for c in range(3):
                        patch = cube_res["cropped"][r*ch:(r+1)*ch, c*cw:(c+1)*cw]
                        raw_bgrs.append(np.median(patch, axis=(0,1)).astype(np.uint8))
                
                # 3. Success with Hybrid mode
                det[4] = expected_center  # Anchor the center sticker
                overlay = cv2.cvtColor(cube_res["annotated"], cv2.COLOR_BGR2RGB)
                return det, raw_bgrs, overlay, None
            
            # Diagnostic Overlay (if no cube and no 9 stickers)
            diag_bgr, _ = yolo_detect.detect_and_draw(raw_bytes)
            diag_rgb = cv2.cvtColor(diag_bgr, cv2.COLOR_BGR2RGB)
            msg = f"⚠️ YOLO detected {len(stickers)} features (expected 9). Falling back to diagnostic view."
            return None, None, diag_rgb, msg

        std = get_std_colors()
        det = []
        raw_bgrs = []
        for s in stickers:
            bgr = np.median(s["cropped"], axis=(0, 1)).astype(np.uint8)
            raw_bgrs.append(bgr)
            if s["color"]:
                det.append(s["color"])
            else:
                det.append(classify_color_lab(bgr, std))
        
        # Success with pure YOLO stickers
        annotated_bgr, _ = yolo_detect.detect_and_draw(raw_bytes)
        overlay = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return det, raw_bgrs, overlay, None
        
    except Exception as e:
        return None, None, None, f"⚠️ YOLO Error: {str(e)}"

def run_method_c(raw_bytes, expected_center):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None, None, None, "❌ Cannot decode image."
    std = get_std_colors(); warped = _warp_to_300(img)
    det, raw_bgrs = _grid_colors_with_pixels(warped, std, lambda b: classify_color_svm(b), use_blocks=True)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    overlay = _draw_grid_overlay(warped_rgb)
    return det, raw_bgrs, overlay, None

# ══════════════════════════════════════════════════════════════════════════════
# LIVE CUBE MAP (renders all 6 faces as mini grids via components.html)
# ══════════════════════════════════════════════════════════════════════════════
def render_live_cube_map(active_face):
    """Render an HTML cross-layout cube map showing all 6 faces using components.html."""
    cube = st.session_state.cube_state
    confirmed = st.session_state.confirmed_faces

    def face_html(area_name, face_name):
        is_active = (face_name == active_face)
        is_confirmed = (face_name in confirmed)
        title_color = "#6366f1" if is_active else ("#22c55e" if is_confirmed else "#94a3b8")
        title_weight = "800" if is_active else "700"
        border_color = "#6366f1" if is_active else ("#22c55e" if is_confirmed else "transparent")
        shadow = "0 0 12px rgba(99,102,241,0.3)" if is_active else ("0 0 8px rgba(34,197,94,0.2)" if is_confirmed else "none")
        status_icon = "✏️" if is_active else ("✅" if is_confirmed else "⭕")
        
        cells = ""
        for idx in range(9):
            color = cube[face_name][idx]
            hex_c = HEX_COLORS.get(color, '#f1f5f9')
            cells += f'<div style="width:22px;height:22px;border-radius:3px;background:{hex_c};"></div>'
        
        return f'''<div style="grid-area:{area_name}; justify-self:center;">
            <div style="font-size:10px;font-weight:{title_weight};text-align:center;color:{title_color};margin-bottom:4px;letter-spacing:1px;font-family:Outfit,sans-serif;">{status_icon} {face_name}</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2px;border-radius:8px;overflow:hidden;border:2px solid {border_color};box-shadow:{shadow};padding:2px;background:white;">{cells}</div>
        </div>'''
    
    html = f'''
    <html><body style="margin:0;padding:2px;background:transparent;font-family:Outfit,sans-serif;box-sizing:border-box;">
    <div style="background:rgba(255,255,255,0.9);border-radius:20px;padding:15px;border:1px solid rgba(0,0,0,0.06);box-shadow:0 8px 24px -8px rgba(0,0,0,0.06); width:fit-content; margin:0 auto;">
        <div style="font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#64748b;margin-bottom:15px;text-align:center;">🗺️ LIVE CUBE MAP</div>
        <div style="display:grid; grid-template-areas: '. U . .' 'L F R B' '. D . .'; grid-gap:8px; justify-content:center; align-items:center;">
            {face_html('U', 'Up')}
            {face_html('L', 'Left')}
            {face_html('F', 'Front')}
            {face_html('R', 'Right')}
            {face_html('B', 'Back')}
            {face_html('D', 'Down')}
        </div>
        <div style="text-align:center;margin-top:15px;font-size:10px;color:#94a3b8;letter-spacing:1px;font-family:Outfit,sans-serif;">
            ✅ {len(confirmed)}/6 CONFIRMED
        </div>
    </div>
    </body></html>'''
    
    components.html(html, height=420)


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION FEEDBACK PANEL
# ══════════════════════════════════════════════════════════════════════════════
def render_detection_feedback(scan_result):
    """Show detailed visual feedback of what was detected."""
    if scan_result is None:
        return
    
    det_colors = scan_result.get('detected', [])
    raw_bgrs = scan_result.get('raw_bgrs', [])
    overlay_img = scan_result.get('overlay', None)
    engine = scan_result.get('engine', 'OpenCV')
    face = scan_result.get('face', 'Front')
    
    st.markdown(f"##### 🔍 Detection Result — {face}")
    
    # Show the warped image with grid overlay
    if overlay_img is not None:
        st.image(overlay_img, caption=f"📐 How {engine} cropped & analyzed your photo", use_container_width=True)
    
    st.markdown("---")
    
    # Build detected color grid HTML with inline styles
    grid_style = "display:grid;grid-template-columns:repeat(3,1fr);gap:4px;max-width:180px;margin:0 auto;"
    cell_style_base = "width:52px;height:52px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;border:2px solid rgba(255,255,255,0.6);box-shadow:inset 0 -2px 4px rgba(0,0,0,0.1);"
    
    cells_html = ""
    for idx in range(9):
        if idx < len(det_colors):
            color = det_colors[idx]
            hex_c = HEX_COLORS.get(color, '#f1f5f9')
            text_color = "#333" if color in ['White','Yellow'] else "rgba(255,255,255,0.9)"
            label = color[:3]
            cells_html += f'<div style="{cell_style_base}background:{hex_c};color:{text_color};">{label}</div>'
    
    pixel_html = ""
    for idx in range(9):
        if idx < len(raw_bgrs):
            bgr = raw_bgrs[idx]
            r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
            pixel_html += f'<div style="{cell_style_base}background:rgb({r},{g},{b});font-size:8px;color:rgba(255,255,255,0.85);">R{r}<br>G{g}<br>B{b}</div>'
    
    col_det, col_raw = st.columns(2)
    with col_det:
        st.markdown("**AI Classification:**")
        st.markdown(f'<div style="{grid_style}">{cells_html}</div>', unsafe_allow_html=True)
    with col_raw:
        st.markdown("**Raw Pixel Colors:**")
        st.markdown(f'<div style="{grid_style}">{pixel_html}</div>', unsafe_allow_html=True)
    
    st.markdown("")
    # Quick summary
    color_counts = Counter(det_colors)
    summary_parts = []
    for c in ['White','Red','Green','Yellow','Orange','Blue']:
        cnt = color_counts.get(c, 0)
        if cnt > 0:
            summary_parts.append(f"{COLOR_EMOJIS[c]} {c}×{cnt}")
    st.caption("Detected: " + "  ".join(summary_parts))

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
    if app_mode == "🧩 Scan & Solve":
        with st.expander("📊 Sticker Stats"):
            all_s = [s for f in FACES for s in st.session_state.cube_state[f]]
            for name in HEX_COLORS:
                cnt = all_s.count(name); ok = (cnt==9)
                st.markdown(f"<div style='font-size:11px;'>{COLOR_EMOJIS[name]} {name}: <b>{cnt}/9</b></div>", unsafe_allow_html=True)
        if st.button("🗑️ Reset Cube", use_container_width=True):
            st.session_state.cube_state = {f:(['White']*4+[CENTER_COLORS[f]]+['White']*4)for f in FACES}
            st.session_state.confirmed_faces = []; st.session_state.last_solution = None
            st.session_state.scan_result = None
            push_history(); st.rerun()

# ── Title is now moved to sidebar for a cleaner studio layout ──

# ── MAIN TITLE ──────────────────────────────────────────────────────────────
st.markdown('''
    <div class="app-title">🧊 AI Rubik's Vision Engine</div>
    <div class="app-subtitle">Multi-Algorithm Comparison & Topology Validation System</div>
''', unsafe_allow_html=True)

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
        is_conf = face_complete(f)
        lbl = f"{COLOR_EMOJIS[cc]} {f}"
        if is_conf and not is_act:
            lbl = f"✅ {f}"
        btn_type = "primary" if is_act else "secondary"
        if pw_cols[i].button(lbl, key=f"pwr_{f}", use_container_width=True, type=btn_type):
            st.session_state.active_face = f
            st.session_state.selected_color = cc
            st.session_state.scan_result = None
            st.rerun()

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # ── Main 3-Column Layout: Upload | Grid | Live Map ──────────────────────
    col_l, col_r, col_map = st.columns([3, 2, 2], gap="large")
    
    with col_l:
        st.markdown("#### 📂 Photo Assist")
        up = st.file_uploader("Upload reference", type=['jpg','png','jpeg'], key=f"up_{curr}", label_visibility="collapsed")
        
        if up:
            raw = up.read()
            # 1. Show the uploaded original photo
            st.image(raw, caption="📷 Your uploaded photo", use_container_width=True)
            
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
                    
                    det, raw_bgrs, overlay, err = None, None, None, None
                    
                    # 4. Algorithm routing
                    if "OpenCV" in algo_choice:
                        det, raw_bgrs, overlay, err = run_method_a(raw, CENTER_COLORS[curr])
                        
                    elif "YOLOv8" in algo_choice:
                        det, raw_bgrs, overlay, err = run_method_b(raw, CENTER_COLORS[curr])
                        
                    elif "SVM" in algo_choice:
                        det, raw_bgrs, overlay, err = run_method_c(raw, CENTER_COLORS[curr])

                    if err:
                        st.error(err)
                        # Diagnostic: Still show the overlay if available
                        if overlay is not None:
                            st.session_state.scan_result = {
                                'detected': det if det else [],
                                'raw_bgrs': [b.tolist() for b in raw_bgrs] if raw_bgrs else [],
                                'overlay': overlay, 'engine': engine_name, 'face': curr
                            }
                    elif det:
                        det[4] = CENTER_COLORS[curr]
                        st.session_state.cube_state[curr] = det
                        mark_confirmed(curr); push_history()
                        
                        st.session_state.scan_result = {
                            'detected': det,
                            'raw_bgrs': [bgr.tolist() if hasattr(bgr, 'tolist') else list(bgr) for bgr in raw_bgrs],
                            'overlay': overlay,
                            'engine': engine_name,
                            'face': curr,
                        }
                        st.rerun()
            
            # Show persistent detection feedback (survives rerun)
            if st.session_state.scan_result and st.session_state.scan_result.get('face') == curr:
                sr = st.session_state.scan_result
                # Reconstruct bgr arrays
                bgr_arrays = [np.array(b, dtype=np.uint8) for b in sr['raw_bgrs']]
                render_detection_feedback({
                    'detected': sr['detected'],
                    'raw_bgrs': bgr_arrays,
                    'overlay': sr.get('overlay'),
                    'engine': sr['engine'],
                    'face': sr['face'],
                })
                
                # Accept / Retry buttons
                st.markdown("")
                bc1, bc2 = st.columns(2)
                if bc1.button("✅ Accept & Next Face", type="primary", use_container_width=True):
                    st.session_state.scan_result = None
                    next_idx = (FACES.index(curr)+1) % 6
                    remaining = [f for f in FACES if not face_complete(f)]
                    st.session_state.active_face = remaining[0] if remaining else FACES[next_idx]
                    st.rerun()
                if bc2.button("🔄 Retry Scan", use_container_width=True):
                    unmark_confirmed(curr)
                    st.session_state.cube_state[curr] = ['White']*4+[CENTER_COLORS[curr]]+['White']*4
                    st.session_state.scan_result = None
                    push_history(); st.rerun()
        else:
            # No file uploaded — show helpful hint
            st.info(f"📷 Upload a photo of the **{curr}** face (center = {COLOR_EMOJIS[CENTER_COLORS[curr]]} {CENTER_COLORS[curr]})")
    
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
    
    with col_map:
        render_live_cube_map(curr)

    # Action Footer
    st.markdown('<div class="action-row">', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    if a1.button("🧹 Reset", use_container_width=True):
        st.session_state.cube_state[curr] = ['White']*4+[CENTER_COLORS[curr]]+['White']*4
        unmark_confirmed(curr); st.session_state.scan_result = None
        push_history(); st.rerun()
    if a2.button("🎨 Fill", use_container_width=True):
        sel = st.session_state.selected_color
        st.session_state.cube_state[curr] = [sel]*4+[CENTER_COLORS[curr]]+[sel]*4
        mark_confirmed(curr); push_history(); st.rerun()
    if a3.button("🚀 Confirm Face", use_container_width=True, type="primary"):
        mark_confirmed(curr); rem = [f for f in FACES if not face_complete(f)]
        if rem: st.session_state.active_face = rem[0]
        st.session_state.scan_result = None
        st.rerun()
    st.markdown('', unsafe_allow_html=True)

    # Result Section
    all_s = [s for f in FACES for s in st.session_state.cube_state[f]]
    errs = [c for c in HEX_COLORS if all_s.count(c)!=9]
    if not errs:
        st.success("✨ Ready to solve!"); 
        if st.button("⚡ Solve Cube", use_container_width=True, type="primary"):
            sol = solve_cube(st.session_state.cube_state)
            if sol.startswith("!"): st.error(sol)
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