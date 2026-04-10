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

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    font-family:'Outfit',sans-serif!important;
    background:#ffffff!important; color:#1e293b!important;}
[data-testid="stMainBlockContainer"]{padding-top:20px!important;}

/* ── Minimalist Card ── */
.mcard{background:#ffffff; border:1.5px solid #e2e8f0; border-radius:14px;
    padding:20px; margin-bottom:16px; box-shadow:0 4px 20px rgba(0,0,0,.04);}
.slabel{font-size:10px; font-weight:700; letter-spacing:1px; text-transform:uppercase;
    color:#94a3b8; margin-bottom:8px; display:block;}

/* ── Unified Command Bar ── */
.cmd-bar{display:flex; gap:6px; margin-bottom:12px; flex-wrap:wrap;}
.cmd-chip{flex:1; min-width:60px; background:#f8fafc; border-radius:8px; padding:6px 4px;
    text-align:center; font-size:10px; font-weight:700; border:1.5px solid #e2e8f0; cursor:pointer;}
.cmd-active{border-color:#6366f1; background:#f5f3ff; color:#6366f1;}

/* ── Action buttons ── */
.action-row{display:flex; gap:8px; margin-top:16px; padding-top:16px; border-top:1px solid #f1f5f9;}
.stButton>button{border-radius:10px!important; font-family:'Outfit',sans-serif!important;
    font-weight:600!important; transition:all .15s ease!important;}

/* ── Solution box ── */
.sol-box{background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:16px;
    font-family:'Courier New',monospace; font-size:14px; font-weight:700; color:#1e293b;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:#f8fafc!important; border-right:1px solid #e2e8f0!important;}
[data-testid="stSidebar"] *{color:#475569!important;}
[data-testid="stSidebar"] hr{border-color:#e2e8f0!important;}
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
    detected = _grid_colors(warped, std, lambda b: classify_color_lab(b, std))
    return detected, cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), None

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
            push_history(); st.rerun()

st.title("🧩 Rubik's Solver")
st.markdown("<p style='color:#94a3b8; margin-top:-15px; margin-bottom:25px;'>Concise Computer Vision AI</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCAN & SOLVE PAGE
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "🧩 Scan & Solve":
    curr = st.session_state.active_face
    st.markdown('<div class="mcard">', unsafe_allow_html=True)
    
    # Nav & Tool Bar
    st.markdown('<span class="slabel">📍 Navigation & Brush</span>', unsafe_allow_html=True)
    c1 = st.columns(6)
    for i, f in enumerate(FACES):
        if c1[i].button(f"{COLOR_EMOJIS[CENTER_COLORS[f]]}\n{f}", key=f"n_{f}", use_container_width=True, type="primary" if f==curr else "secondary"):
            st.session_state.active_face = f; st.session_state.selected_color = CENTER_COLORS[f]; st.rerun()
            
    sel = st.session_state.selected_color
    c2 = st.columns(6)
    for i, cname in enumerate(HEX_COLORS):
        if c2[i].button(f"{'✅' if sel==cname else ''}{COLOR_EMOJIS[cname]}", key=f"p_{cname}", use_container_width=True):
            st.session_state.selected_color = cname; st.rerun()

    st.divider()

    # Input Body
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<span class="slabel">📷 Scan Face</span>', unsafe_allow_html=True)
        up = st.file_uploader("Upload", type=['jpg','png','jpeg'], key=f"up_{curr}", label_visibility="collapsed")
        if up:
            raw = up.read(); st.image(raw, use_container_width=True)
            if st.button("🔍 Run Magic Scan", use_container_width=True, type="primary"):
                with st.spinner("Analyzing..."):
                    det, img, err = run_method_a(raw, CENTER_COLORS[curr])
                    if err: st.error(err)
                    else:
                        st.session_state.cube_state[curr] = det; det[4] = CENTER_COLORS[curr]
                        mark_confirmed(curr); push_history(); st.rerun()
    with col_r:
        st.markdown('<span class="slabel">✏️ Manual Edit</span>', unsafe_allow_html=True)
        def pnt(face, ix): st.session_state.cube_state[face][ix] = st.session_state.selected_color; mark_confirmed(face); push_history()
        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r*3+c; cv = st.session_state.cube_state[curr][idx]
                if idx==4: cols[c].button(f"🔒{COLOR_EMOJIS[cv]}", disabled=True, use_container_width=True)
                else: cols[c].button(f"{COLOR_EMOJIS[cv]}", key=f"g_{curr}_{idx}", on_click=pnt, args=(curr, idx), use_container_width=True)

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
    st.markdown('</div></div>', unsafe_allow_html=True)

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
