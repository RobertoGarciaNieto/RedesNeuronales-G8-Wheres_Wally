"""
Where's Waldo — Frontend v3.6 (Canvas de Arrastre Final)
Grupo 8 · Redes Neuronales
"""

import base64
import io
import math
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO

from utils import calcular_iou

SCRIPT_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Where's Waldo · IA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTES GLOBALIZADAS
# ─────────────────────────────────────────────
CLASS_NAMES = {0: "odlaw", 1: "waldo", 2: "wilma", 3: "wizard", 4: "woof"}
CLASS_COLORS = {
    "waldo": "#FF3333", "odlaw": "#FFD700", "wilma": "#33CC44",
    "wizard": "#3399FF", "woof": "#FF9900",
}
CLASS_COLORS_PIL = {
    "waldo": (255,51,51), "odlaw": (255,215,0), "wilma": (51,204,68),
    "wizard": (51,153,255), "woof": (255,153,0),
}
CLASS_EMOJI = {
    "waldo": "🔴", "odlaw": "🟡", "wilma": "🟢",
    "wizard": "🔵", "woof": "🟠",
}
MAX_DISPLAY_WIDTH = 1350
NMS_IOU_THRESHOLD = 0.35
ZOOM_HEIGHT = 650

# ─────────────────────────────────────────────
# ESTILOS CSS PERSONALIZADOS
# ─────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bangers&family=Nunito:wght@400;600;700;800&display=swap');
footer {visibility: hidden;}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%) !important;
    border-right: 4px solid #e63946 !important;
}
[data-testid="stButton"] > button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important; border-radius: 10px !important;
    transition: transform .12s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(230,57,70,.35) !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #3399FF !important;
    background: rgba(51, 153, 255, 0.05) !important;
    border-radius: 15px !important;
    padding: 30px !important;
}
/* Canvas ocupa todo el ancho del contenedor */
canvas.upper-canvas, canvas.lower-canvas {
    max-width: 100% !important;
}
.canvas-container {
    max-width: 100% !important;
    width: 100% !important;
}
/* Maximizar el contenedor de bloques de Streamlit para el Canvas */
[data-testid="stVerticalBlock"] > div:has(canvas) {
    width: 100% !important;
    max-width: 100% !important;
}
</style>
""")

# ─────────────────────────────────────────────
# CORE DE REDES NEURONALES (Inferencia y SAHI)
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    try:
        model_path = SCRIPT_DIR / "best.pt"
        if not model_path.exists(): return None
        return YOLO(str(model_path))
    except:
        return None

def nms_por_clase(dets, iou_thresh=NMS_IOU_THRESHOLD):
    if not dets: return []
    resultado = []
    for cls in set(d["cls"] for d in dets):
        bucket = sorted([d for d in dets if d["cls"]==cls], key=lambda d: d["conf"], reverse=True)
        kept = []
        while bucket:
            best = bucket.pop(0)
            kept.append(best)
            bucket = [d for d in bucket if calcular_iou(
                (best["x1"],best["y1"],best["x2"],best["y2"]),
                (d["x1"],d["y1"],d["x2"],d["y2"])) < iou_thresh]
        resultado.extend(kept)
    return resultado

def inferencia_sahi(modelo, img, patch=640, overlap=0.25):
    w, h = img.size
    step = int(patch * (1 - overlap))
    dets = []
    def offsets(dim):
        offs = list(range(0, dim - patch + 1, step))
        if not offs or offs[-1] + patch < dim: offs.append(max(0, dim - patch))
        return offs
    total = len(offsets(w)) * len(offsets(h))
    counter = [0]
    progress = st.progress(0, text="🔎 Analizando parches con SAHI...")
    for ox in offsets(w):
        for oy in offsets(h):
            parche = img.crop((ox, oy, min(ox+patch, w), min(oy+patch, h)))
            for r in modelo(parche, conf=0.50, verbose=False):
                for box in r.boxes:
                    bx1,by1,bx2,by2 = box.xyxy[0].tolist()
                    dets.append({"x1":bx1+ox,"y1":by1+oy,"x2":bx2+ox,"y2":by2+oy,
                                 "conf":float(box.conf[0]),"cls":int(box.cls[0])})
            counter[0] += 1
            progress.progress(counter[0]/total, text=f"🔎 Parche {counter[0]}/{total}")
    progress.empty()
    return nms_por_clase(dets)

def anotar_imagen(img_pil, dets):
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 20)
    except: font = ImageFont.load_default()
    for d in dets:
        nombre = CLASS_NAMES.get(d["cls"],"?")
        color = CLASS_COLORS_PIL.get(nombre,(255,255,255))
        x1,y1,x2,y2 = int(d["x1"]),int(d["y1"]),int(d["x2"]),int(d["y2"])
        for t in range(4): draw.rectangle([x1-t,y1-t,x2+t,y2+t], outline=color)
        lbl = f"{nombre.upper()} {d['conf']:.0%}"
        bb = draw.textbbox((x1,y1-26), lbl, font=font)
        draw.rectangle(bb, fill=color)
        draw.text((x1,y1-26), lbl, fill="black", font=font)
    return img

@st.cache_data
def cargar_ground_truth():
    """Carga las coordenadas reales (oficiales) del dataset de test.
    Busca el archivo tanto al lado de app.py como dentro de posters_fijos/."""
    posibles_rutas = [
        SCRIPT_DIR / "ground_truth.json",
        SCRIPT_DIR / "posters_fijos" / "ground_truth.json",
    ]
    for gt_path in posibles_rutas:
        if gt_path.exists():
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
    return None

def evaluar_contra_ground_truth(candidatos, gt_list, tol_iou):
    """
    Compara candidatos contra el ground truth.
    También detecta si un candidato coincide espacialmente con un GT de DISTINTO personaje
    (el usuario marcó la zona correcta pero con el personaje equivocado).
    """
    resultados = []
    for gt in gt_list:
        gt_box = tuple(gt["coords"])
        mejor_iou = 0.0
        mejor_candidato = None
        personaje_equivocado = None  # nombre del personaje que el usuario marcó en esa zona

        for c in candidatos:
            iou = calcular_iou(c["coords"], gt_box)
            if c["personaje"] == gt["personaje"]:
                if mejor_candidato is None or iou > mejor_iou:
                    mejor_iou = iou
                    mejor_candidato = c["coords"]
            elif iou >= tol_iou and personaje_equivocado is None:
                # Marcó esta zona pero con el personaje equivocado
                personaje_equivocado = c["personaje"]

        resultados.append({
            "personaje": gt["personaje"],
            "encontrado": mejor_iou >= tol_iou,
            "iou": mejor_iou,
            "candidato": mejor_candidato,
            "gt_box": gt_box,
            "personaje_equivocado": personaje_equivocado,
        })
    return resultados

def feedback_cualitativo(resultado, tol_iou):
    """Da una pista textual de qué tan cerca estuvo un candidato del ground truth."""
    if resultado["encontrado"]:
        return "¡Encontrado!"

    # Marcó la zona correcta pero con el personaje equivocado
    if resultado.get("personaje_equivocado"):
        p = resultado["personaje_equivocado"].upper()
        import random
        return random.choice([
            f"Encontraste la zona, pero ese no es {p}. ¡Cambiá el personaje!",
            f"¡Estás ahí! Pero eso que marcaste como {p} no es correcto.",
            f"La zona es correcta, pero no es {p}. Mirá bien quién está ahí.",
        ])

    if resultado["candidato"] is None:
        return "No marcaste a este personaje."

    ux1, uy1, ux2, uy2 = resultado["candidato"]
    gx1, gy1, gx2, gy2 = resultado["gt_box"]
    cx, cy = (ux1 + ux2) / 2, (uy1 + uy2) / 2
    centro_dentro = gx1 <= cx <= gx2 and gy1 <= cy <= gy2

    if resultado["iou"] > 0 or centro_dentro:
        return f"Estás en la zona correcta (IoU {resultado['iou']:.2f}), pero el recuadro no encaja bien. Ajustalo un poco."
    return "Lejos de la posición real — revisá otra zona del póster."

MENSAJES_CONFUSION = [
    "Ese no es {p}, ¡seguí buscando!",
    "Mmm... ese no parece {p}. Mirá con más atención.",
    "¡Casi! Pero ese no es {p}. No te rindas.",
    "Ese no es {p}... ¿seguro que lo estás viendo bien?",
    "Nope, ese no es {p}. ¡El disfraz te confundió!",
    "¡Cuidado! Ese no es {p}. Hay mucha gente parecida.",
    "Ese no es {p}. Dale otra mirada al póster.",
]

def detectar_confusion_personaje(rectangulos, gt_list, tol_iou):
    """
    Para cada rectángulo del usuario, revisa si coincide espacialmente con un GT
    de DISTINTO personaje → mensaje genérico sin revelar quién es realmente.
    """
    import random
    mensajes = []
    for (ux1, uy1, ux2, uy2, p_usuario) in rectangulos:
        for gt in gt_list:
            if gt["personaje"] == p_usuario:
                continue
            gx1, gy1, gx2, gy2 = gt["coords"]
            iou = calcular_iou((ux1, uy1, ux2, uy2), (gx1, gy1, gx2, gy2))
            if iou >= tol_iou:
                plantilla = random.choice(MENSAJES_CONFUSION)
                msg = plantilla.format(p=p_usuario.upper())
                mensajes.append((p_usuario, msg))
                break
    return mensajes

def anotar_resultado_juego(img_pil, res_usuario, res_ia):
    """Dibuja sobre el póster completo: la posición real (dorado), tu recuadro (rojo)
    y el de la IA (azul), para poder comparar visualmente."""
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 22)
    except: font = ImageFont.load_default()

    for ru, ri in zip(res_usuario, res_ia):
        gx1, gy1, gx2, gy2 = ru["gt_box"]
        for t in range(3):
            draw.rectangle([gx1-t, gy1-t, gx2+t, gy2+t], outline=(255, 215, 0))
        lbl = f"REAL: {ru['personaje'].upper()}"
        bb = draw.textbbox((gx1, gy1 - 26), lbl, font=font)
        draw.rectangle(bb, fill=(255, 215, 0))
        draw.text((gx1, gy1 - 26), lbl, fill="black", font=font)

        if ru["candidato"] is not None:
            ux1, uy1, ux2, uy2 = ru["candidato"]
            for t in range(2):
                draw.rectangle([ux1-t, uy1-t, ux2+t, uy2+t], outline=(230, 57, 70))

        if ri["candidato"] is not None:
            ix1, iy1, ix2, iy2 = ri["candidato"]
            for t in range(2):
                draw.rectangle([ix1-t, iy1-t, ix2+t, iy2+t], outline=(51, 153, 255))

    return img

def pil_b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def render_zoom_viewer(img_b64, dw, dh):
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:transparent;overflow:hidden;}}
#wrap{{position:relative;width:100%;height:{ZOOM_HEIGHT}px;overflow:hidden;background:#0d0d0d;cursor:grab;border-radius:12px;border:2px solid #e63946;}}
canvas{{position:absolute;top:0;left:0;transform-origin:0 0;}}
#ctrls{{position:absolute;top:10px;left:10px;display:flex;gap:6px;z-index:99;}}
.b{{width:34px;height:34px;background:rgba(20,20,20,.85);border:1px solid #e63946;color:#fff;font-size:17px;border-radius:7px;cursor:pointer;display:flex;align-items:center;justify-content:center;user-select:none;}}
.b:hover{{background:#e63946;}}
#ind{{position:absolute;top:10px;right:10px;background:rgba(0,0,0,.75);color:#fff;padding:3px 10px;border-radius:20px;font:700 12px/1 Arial;z-index:99;border:1px solid #555;}}
#bar{{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.65);color:#999;font:11px Arial;padding:4px 10px;z-index:99;}}
</style></head><body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="ctrls">
    <div class="b" onclick="zm(0.3)">+</div>
    <div class="b" onclick="zm(-0.3)">−</div>
    <div class="b" onclick="reset()" style="font-size:13px">⟳</div>
  </div>
  <div id="ind">1.0×</div>
  <div id="bar">🔍 Solo exploración — Rueda: zoom · Arrastrar: mover</div>
</div>
<script>
const OW={dw},OH={dh},CH={ZOOM_HEIGHT};
const w=document.getElementById('wrap'),c=document.getElementById('c'),ctx=c.getContext('2d'),ind=document.getElementById('ind');
let sc=1,px=0,py=0,dr=false,x0=0,y0=0,px0=0,py0=0;
const im=new Image(); im.src='data:image/jpeg;base64,{img_b64}';
im.onload=()=>{{c.width=OW;c.height=OH;ctx.drawImage(im,0,0,OW,OH);fitImg();}};
function fitImg() {{const cw=w.clientWidth;sc=Math.min(cw/OW,CH/OH);px=(cw-OW*sc)/2;py=(CH-OH*sc)/2;ap();}}
function ap() {{c.style.transform=`translate(${{px}}px,${{py}}px) scale(${{sc}})`;ind.textContent=sc.toFixed(1)+'×';}}
function cl() {{const cw=w.clientWidth,iw=OW*sc,ih=OH*sc;px=iw<=cw?(cw-iw)/2:Math.min(0,Math.max(px,cw-iw));py=ih<=CH?(CH-ih)/2:Math.min(0,Math.max(py,CH-ih));}}
function zm(d,cx,cy){{cx=cx??w.clientWidth/2;cy=cy??CH/2;const ns=Math.min(8,Math.max(0.5,sc+d)),r=ns/sc;px=cx-r*(cx-px);py=cy-r*(cy-py);sc=ns;cl();ap();}}
function reset(){{fitImg();}}
w.addEventListener('wheel',e=>{{e.preventDefault();const r=w.getBoundingClientRect();zm(e.deltaY<0?.25:-.25,e.clientX-r.left,e.clientY-r.top);}},{{passive:false}});
w.addEventListener('mousedown',e=>{{if(e.button!==0)return;dr=true;x0=e.clientX;y0=e.clientY;px0=px;py0=py;w.style.cursor='grabbing';}});
window.addEventListener('mousemove',e=>{{if(!dr)return;px=px0+(e.clientX-x0);py=py0+(e.clientY-y0);cl();ap();}});
window.addEventListener('mouseup',()=>{{dr=false;w.style.cursor='grab';}});
</script></body></html>"""
    components.html(html, height=ZOOM_HEIGHT+12, scrolling=False)

def obtener_detecciones(modelo, img_pil, img_key):
    if "dets_cache" not in st.session_state: st.session_state.dets_cache = {}
    if img_key in st.session_state.dets_cache: return st.session_state.dets_cache[img_key]
    dets = inferencia_sahi(modelo, img_pil)
    st.session_state.dets_cache[img_key] = dets
    return dets

# ─────────────────────────────────────────────
# INTERFAZ LATERAL (Mantenemos tu Sidebar Favorito)
# ─────────────────────────────────────────────
def render_sidebar_juego():
    st.markdown('<p style="color:#c9d1d9;font-family:Nunito;font-weight:700;margin-bottom:8px;">🖊️ Tocá un personaje y marcalo en el póster</p>', unsafe_allow_html=True)
    chars = list(CLASS_NAMES.values())
    for i in range(0, len(chars), 2):
        cols = st.columns(2)
        for col, nombre in zip(cols, chars[i:i+2]):
            activo = st.session_state.personaje_sel == nombre
            with col:
                if st.button(f"{CLASS_EMOJI[nombre]} {nombre.upper()}", key=f"side_ch_{nombre}", use_container_width=True, type="primary" if activo else "secondary"):
                    st.session_state.personaje_sel = nombre
                    st.rerun()
    psel = st.session_state.personaje_sel
    csel = CLASS_COLORS[psel]
    st.markdown(f'<div style="margin:8px 0;padding:6px 0;text-align:center;border-top:3px solid {csel};border-bottom:3px solid {csel};"><span style="color:{csel};font-family:Nunito;font-weight:800;font-size:.9rem;">HERRAMIENTA: {psel.upper()}</span></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<p style="color:#c9d1d9;font-family:Nunito;font-weight:700;margin-bottom:4px;">🎯 Tolerancia IoU para validar</p>', unsafe_allow_html=True)
    st.session_state.tolerancia_iou = st.slider("tol", 0.10, 0.90, 0.50, 0.05, label_visibility="collapsed")

def inicializar_estado():
    defaults = {
        "page": "Inicio", "personaje_sel": "waldo", "tolerancia_iou": 0.50,
        "dets_cache": {}, "imagen_bytes": None, "imagen_key": None,
        "mostrar_solucion": False, "rectangulos": [], "resultados_verificacion": None,
        "mostrar_comparacion": False
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# ─────────────────────────────────────────────
# FLUJO PRINCIPAL NAVEGABLE
# ─────────────────────────────────────────────
def main():
    inicializar_estado()

    with st.sidebar:
        st.markdown("<h2 style='text-align:center; color:#fff; font-family:Bangers; letter-spacing:2px; text-shadow:2px 2px 0 #e63946;'>🔍 WALDO IA</h2>", unsafe_allow_html=True)
        st.divider()
        if st.button("🏠 Inicio y Reglas", use_container_width=True): st.session_state.page = "Inicio"; st.rerun()
        if st.button("🎯 Jugar contra la IA", use_container_width=True): st.session_state.page = "Jugar"; st.session_state.imagen_bytes = None; st.rerun()
        if st.button("🤖 Resolver con IA", use_container_width=True): st.session_state.page = "Resolver"; st.session_state.imagen_bytes = None; st.rerun()

    st.markdown("""
    <div style="background:repeating-linear-gradient(0deg,#e63946 0,#e63946 14px,#fff 14px,#fff 28px);
      border-radius:14px;padding:20px 28px;margin-bottom:30px;box-shadow:0 4px 18px rgba(230,57,70,.2);">
      <span style="font-family:'Bangers',cursive;font-size:2.8rem;color:#fff;letter-spacing:3px;text-shadow:3px 3px 0 #1d3557,-1px -1px 0 #1d3557;">🔍 WHERE'S WALDO · IA</span>
    </div>""", unsafe_allow_html=True)

    modelo = cargar_modelo()

    # 🏠 PÁGINA: INICIO Y REGLAS (Introducción + paso a paso + acceso a los dos modos)
    if st.session_state.page == "Inicio":
        st.markdown("""
        <div style="background: linear-gradient(145deg, #1a1a2e, #16213e); padding: 28px 32px; border-radius: 15px; border: 2px solid #3399FF; margin-bottom: 22px;">
            <h2 style="font-family:'Bangers'; color:#3399FF; font-size:2.1rem; margin-top:0;">📖 ¿De qué se trata?</h2>
            <p style="color:#c9d1d9; font-family:'Nunito'; font-size:1.02rem; line-height:1.6; margin-bottom:0;">
                Entrenamos una red neuronal (YOLO) para encontrar a los personajes clásicos de "¿Dónde está Wally?"
                — <b>Waldo</b>, <b>Wilma</b>, <b>Odlaw</b>, el <b>Mago</b> y <b>Woof</b> — dentro de pósters repletos de gente.
                Acá podés <b>competir contra la IA</b> tratando de encontrarlos vos mismo, o <b>subir tu propio póster</b>
                para que la IA lo resuelva por vos.
            </p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; font-family:Bangers; color:#e63946; font-size:2.5rem; margin-bottom:20px;'>¡ELEGÍ TU DESAFÍO!</h2>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("""<div style="background: linear-gradient(145deg, #1a1a2e, #16213e); padding: 22px 24px; border-radius: 14px; border: 2px solid #e63946; height:400px;">
                <h3 style="color:#e63946; font-family:Bangers; font-size:1.8rem;">🎯 JUGAR CONTRA LA IA</h3>
                <p style="color:#c9d1d9; font-family:Nunito;">Compite en tiempo real arrastrando el mouse para dibujar recuadros.</p>
                <ul style="color:#c9d1d9; font-family:'Nunito'; font-size:.92rem; line-height:1.7; padding-left:18px; margin-bottom:0;">
                    <li>Elegís uno de 3 pósters fijos (Fácil, Medio, Difícil) de nuestro dataset de test.</li>
                    <li>Vos dibujás recuadros sobre el póster marcando dónde creés que está cada personaje.</li>
                    <li>En paralelo, la IA analiza el mismo póster con su propio modelo.</li>
                    <li>Ambos resultados se comparan contra las coordenadas <i>reales</i> del dataset (ground truth).</li>
                    <li>Gana quien acierte más personajes con un IoU igual o mayor a la tolerancia elegida.</li>
                </ul>
            </div>""", unsafe_allow_html=True)
            if st.button("🚀 Comenzar Partida", use_container_width=True): st.session_state.page = "Jugar"; st.rerun()
        with col_r2:
            st.markdown("""<div style="background: linear-gradient(145deg, #1a1a2e, #16213e); padding: 22px 24px; border-radius: 14px; border: 2px solid #3399FF; height: 400px;">
                <h3 style="color:#3399FF; font-family:Bangers; font-size:1.8rem;">🤖 RESOLVER CON IA</h3>
                <p style="color:#c9d1d9; font-family:Nunito;">Sube tu plano general y deja que la Red Neuronal encuentre los personajes.</p>
                <ul style="color:#c9d1d9; font-family:'Nunito'; font-size:.92rem; line-height:1.7; padding-left:18px; margin-bottom:0;">
                    <li>Subís cualquier póster propio, no hace falta que pertenezca al dataset.</li>
                    <li>La IA recorre la imagen en parches superpuestos (SAHI) para no perderse personajes chicos.</li>
                    <li>Te devuelve los recuadros encontrados junto a su nivel de confianza.</li>
                    <li>Solo participa la IA: no hay puntaje ni comparación, es un modo "me rindo" para resolver tu póster.</li>
                </ul>
            </div>""", unsafe_allow_html=True)
            if st.button("🧠 Abrir Resolutor", use_container_width=True): st.session_state.page = "Resolver"; st.rerun()

    # 🎯 PÁGINA: JUGAR VS IA (CARGA AUTOMÁTICA EN CANVAS)
    elif st.session_state.page == "Jugar":
        # ── Personajes a Buscar ──
        CHARS_ORDEN = ["waldo", "wilma", "odlaw", "wizard", "woof"]
        st.markdown("<h3 style='font-family:Bangers;color:#e63946;font-size:2.3rem;text-align:center;letter-spacing:2px;margin-bottom:16px;'>🎭 PERSONAJES A BUSCAR</h3>", unsafe_allow_html=True)
        cols_chars = st.columns(5)
        for col, nombre in zip(cols_chars, CHARS_ORDEN):
            img_path = SCRIPT_DIR / "personajesABuscar" / f"{nombre}.jpg"
            color = CLASS_COLORS.get(nombre, "#fff")
            with col:
                if img_path.exists():
                    img_char = Image.open(img_path)
                    buf_c = io.BytesIO()
                    img_char.save(buf_c, format="JPEG")
                    b64_c = base64.b64encode(buf_c.getvalue()).decode()
                    st.markdown(f"""<div style="border:3px solid {color};border-radius:12px;overflow:hidden;background:#ffffff;text-align:center;"><img src="data:image/jpeg;base64,{b64_c}" style="width:100%;display:block;pointer-events:none;"><div style="font-family:'Bangers';color:{color};font-size:1.1rem;letter-spacing:1px;padding:6px 0 8px;background:#ffffff;">{nombre.upper()}</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="border:3px solid {color};border-radius:12px;background:#ffffff;text-align:center;padding:30px 10px;"><div style="font-size:2.5rem;">{CLASS_EMOJI.get(nombre,'❓')}</div><div style="font-family:'Bangers';color:{color};font-size:1.1rem;letter-spacing:1px;padding-bottom:6px;">{nombre.upper()}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<h2 style='font-family:Bangers;color:#e63946;font-size:2.3rem;'>🎯 Seleccioná el nivel y marcá los personajes</h2>", unsafe_allow_html=True)
        nivel = st.selectbox("Elegí la dificultad:", ["Fácil", "Medio", "Difícil", "Super Dificil"])
        archivo_nombre = {
            "Fácil": "facil.jpg",
            "Medio": "medio.jpg",
            "Difícil": "dificil.jpg",
            "Super Dificil": "superDificil.webp",
        }.get(nivel, "facil.jpg")
        path_poster = SCRIPT_DIR / "posters_fijos" / archivo_nombre

        if not path_poster.exists():
            st.error(f"⚠️ Poné los pósters fijos en la ruta: `posters_fijos/{archivo_nombre}`")
            return

        if st.session_state.imagen_key != f"juego_{nivel}":
            st.session_state.imagen_key = f"juego_{nivel}"
            with open(path_poster, "rb") as f: st.session_state.imagen_bytes = f.read()
            st.session_state.rectangulos = []
            st.session_state.resultados_verificacion = None
            st.session_state.mostrar_comparacion = False

        with st.sidebar:
            st.divider()
            render_sidebar_juego()

        img_pil = Image.open(io.BytesIO(st.session_state.imagen_bytes)).convert("RGB")
        iw, ih = img_pil.size

        # Escalado adaptado: Aumentamos a 1400px para maximizar el ancho del lienzo
        MAX_CANVAS_WIDTH = 1400
        factor = iw / MAX_CANVAS_WIDTH if iw > MAX_CANVAS_WIDTH else 1.0
        dw, dh = int(iw / factor), int(ih / factor)

        # Imagen para la lupa calculada con su respectiva resolución HD
        factor_zoom = iw / MAX_DISPLAY_WIDTH if iw > MAX_DISPLAY_WIDTH else 1.0
        dw_zoom, dh_zoom = int(iw / factor_zoom), int(ih / factor_zoom)
        img_zoom = img_pil.resize((dw_zoom, dh_zoom), Image.LANCZOS)
        b64 = pil_b64(img_zoom)

        img_display_base = img_pil.resize((dw, dh), Image.LANCZOS)

        with st.expander("🔍 Lupa de Zoom (Solo Exploración de Lectura)", expanded=False):
            render_zoom_viewer(b64, dw_zoom, dh_zoom)

        # 🚀 CANVAS PROFESIONAL (Estructura de Fabric.js corregida para producción)
        buf_canvas = io.BytesIO()
        img_pil.resize((dw, dh), Image.LANCZOS).save(buf_canvas, format="JPEG", quality=85)
        b64_canvas = base64.b64encode(buf_canvas.getvalue()).decode()
        data_url = f"data:image/jpeg;base64,{b64_canvas}"

        # Estructura JSON oficial que Fabric.js espera para renderizar una imagen de fondo legítima
        initial_drawing = {
            "version": "4.4.0",
            "objects": [],
            "backgroundImage": {
                "type": "image",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": 0,
                "top": 0,
                "width": dw,
                "height": dh,
                "src": data_url,
                "scaleX": 1,
                "scaleY": 1
            }
        }

        canvas_result = st_canvas(
            fill_color="rgba(230, 57, 70, 0.15)",
            stroke_width=3,
            stroke_color=CLASS_COLORS[st.session_state.personaje_sel],
            background_color="",
            background_image=None,
            initial_drawing=initial_drawing,
            update_streamlit=True,
            height=dh, width=dw,
            drawing_mode="rect",
            key=f"canvas_juego_{nivel}",
        )

        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            user_rects = []
            
            # Dibujamos las etiquetas sobre una copia de la imagen base
            img_con_etiquetas = img_display_base.copy()
            draw = ImageDraw.Draw(img_con_etiquetas)
            try: font = ImageFont.truetype("arial.ttf", 16)
            except: font = ImageFont.load_default()

            for i, obj in enumerate(objects, start=1):
                if obj["type"] == "rect":
                    # Calcular coordenadas reales
                    scale_x = obj.get("scaleX", 1) or 1
                    scale_y = obj.get("scaleY", 1) or 1
                    w, h = obj["width"] * scale_x, obj["height"] * scale_y
                    x1, y1 = int(obj["left"]), int(obj["top"])
                    x2, y2 = int(x1 + w), int(y1 + h)
                    
                    # Recuperar el personaje por el color del trazo (Hack de colores)
                    color_hex = obj.get("stroke", "#FF3333").upper()
                    personaje = next((k for k, v in CLASS_COLORS.items() if v.upper() == color_hex), "waldo")
                    
                    # Guardar para la evaluación
                    user_rects.append((x1 * factor, y1 * factor, x2 * factor, y2 * factor, personaje))
                    
                    # Dibujar etiqueta #ID PERSONAJE
                    lbl = f"#{i} {personaje.upper()}"
                    bb = draw.textbbox((x1, y1 - 20), lbl, font=font)
                    draw.rectangle(bb, fill=color_hex)
                    draw.text((x1, y1 - 20), lbl, fill="white", font=font)
            st.session_state.rectangulos = user_rects

        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b2:
            if st.button("✅ Verificar recuadros marcados", type="primary", use_container_width=True, disabled=len(st.session_state.rectangulos)==0):
                gt_data = cargar_ground_truth()
                if gt_data is None or nivel not in gt_data:
                    st.error("⚠️ No se encontró `ground_truth.json` con las coordenadas oficiales para este nivel.")
                elif modelo is None:
                    st.error("⚠️ No se detectó el modelo `best.pt` en el directorio raíz.")
                else:
                    gt_list = gt_data[nivel]
                    dets = obtener_detecciones(modelo, img_pil, st.session_state.imagen_key)
                    cand_usuario = [{"coords": (x1, y1, x2, y2), "personaje": p}
                                     for (x1, y1, x2, y2, p) in st.session_state.rectangulos]
                    cand_ia = [{"coords": (d["x1"], d["y1"], d["x2"], d["y2"]), "personaje": CLASS_NAMES.get(d["cls"], "?")}
                               for d in dets]
                    res_usuario = evaluar_contra_ground_truth(cand_usuario, gt_list, st.session_state.tolerancia_iou)
                    res_ia = evaluar_contra_ground_truth(cand_ia, gt_list, st.session_state.tolerancia_iou)
                    st.session_state.resultados_verificacion = {"usuario": res_usuario, "ia": res_ia, "total": len(gt_list)}

        # ── Mensajes de confusión de personaje (punto 2) ──
        if st.session_state.rectangulos and st.session_state.resultados_verificacion is None:
            gt_data = cargar_ground_truth()
            if gt_data and nivel in gt_data:
                confusiones = detectar_confusion_personaje(
                    st.session_state.rectangulos, gt_data[nivel], st.session_state.tolerancia_iou
                )
                for p_user, msg in confusiones:
                    st.markdown(f"""<div style="background:linear-gradient(90deg,#2a1a0e,#1a1a2e);border-left:5px solid #FF9900;
                        border-radius:10px;padding:12px 18px;margin:6px 0;font-family:Nunito;">
                        <span style="font-size:1.3rem;">⚠️</span>
                        <span style="color:#FFB347;font-weight:700;"> {msg}</span>
                    </div>""", unsafe_allow_html=True)

        if st.session_state.resultados_verificacion is not None:
            veredicto = st.session_state.resultados_verificacion
            res_usuario, res_ia, total = veredicto["usuario"], veredicto["ia"], veredicto["total"]
            aciertos_usuario = sum(1 for r in res_usuario if r["encontrado"])
            aciertos_ia = sum(1 for r in res_ia if r["encontrado"])

            if aciertos_usuario > aciertos_ia:
                v_txt, v_color, v_emoji = "¡GANASTE VOS!", "#33CC44", "🏆"
            elif aciertos_ia > aciertos_usuario:
                v_txt, v_color, v_emoji = "GANÓ LA IA", "#e63946", "🤖"
            else:
                v_txt, v_color, v_emoji = "EMPATE", "#FFD700", "🤝"

            # ── Banner veredicto ──
            st.markdown(f"""
            <div style="background:linear-gradient(90deg,#16213e 0%,#1a1a2e 100%);padding:22px 28px;
                border-radius:14px;border-left:6px solid {v_color};margin-top:24px;text-align:center;">
                <div style="font-size:3rem;line-height:1;margin-bottom:6px;">{v_emoji}</div>
                <div style="color:{v_color};font-family:'Bangers';font-size:2.4rem;letter-spacing:2px;">{v_txt}</div>
                <div style="color:#8b949e;font-family:'Nunito';font-size:.9rem;margin-top:6px;">
                    IoU mínimo requerido: <b style="color:#fff;">{st.session_state.tolerancia_iou:.2f}</b>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── Tarjetas de puntaje ──
            st.markdown("<br>", unsafe_allow_html=True)
            col_s1, col_s2 = st.columns(2)
            pct_u = int(aciertos_usuario / total * 100) if total else 0
            pct_ia = int(aciertos_ia / total * 100) if total else 0
            with col_s1:
                st.markdown(f"""<div style="background:linear-gradient(145deg,#1a0a0a,#16213e);padding:20px 24px;
                    border-radius:14px;border:2px solid #e63946;text-align:center;">
                    <div style="color:#e63946;font-family:'Bangers';font-size:1.4rem;letter-spacing:1px;">🧑 VOS</div>
                    <div style="font-size:3rem;font-weight:900;color:#fff;line-height:1.1;margin:8px 0;">
                        {aciertos_usuario}<span style="font-size:1.4rem;color:#8b949e;">/{total}</span>
                    </div>
                    <div style="background:#0d0d0d;border-radius:20px;height:8px;overflow:hidden;margin:8px 0 4px;">
                        <div style="background:#e63946;width:{pct_u}%;height:100%;border-radius:20px;transition:width .5s;"></div>
                    </div>
                    <div style="color:#8b949e;font-family:'Nunito';font-size:.85rem;">{pct_u}% de acierto</div>
                </div>""", unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""<div style="background:linear-gradient(145deg,#0a0f1a,#16213e);padding:20px 24px;
                    border-radius:14px;border:2px solid #3399FF;text-align:center;">
                    <div style="color:#3399FF;font-family:'Bangers';font-size:1.4rem;letter-spacing:1px;">🤖 IA</div>
                    <div style="font-size:3rem;font-weight:900;color:#fff;line-height:1.1;margin:8px 0;">
                        {aciertos_ia}<span style="font-size:1.4rem;color:#8b949e;">/{total}</span>
                    </div>
                    <div style="background:#0d0d0d;border-radius:20px;height:8px;overflow:hidden;margin:8px 0 4px;">
                        <div style="background:#3399FF;width:{pct_ia}%;height:100%;border-radius:20px;transition:width .5s;"></div>
                    </div>
                    <div style="color:#8b949e;font-family:'Nunito';font-size:.85rem;">{pct_ia}% de acierto</div>
                </div>""", unsafe_allow_html=True)

            # ── Grilla de personajes ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='font-family:Bangers;color:#c9d1d9;font-size:1.3rem;margin-bottom:12px;letter-spacing:1px;'>📋 DETALLE POR PERSONAJE</div>", unsafe_allow_html=True)

            for ru, ri in zip(res_usuario, res_ia):
                personaje = ru["personaje"]
                emoji = CLASS_EMOJI.get(personaje, "❓")
                color = CLASS_COLORS.get(personaje, "#fff")
                ok_u = ru["encontrado"]
                ok_i = ri["encontrado"]
                fb_u = feedback_cualitativo(ru, st.session_state.tolerancia_iou)
                fb_i = feedback_cualitativo(ri, st.session_state.tolerancia_iou)
                icon_u = "✅" if ok_u else ("⚠️" if ru.get("personaje_equivocado") else "❌")
                icon_i = "✅" if ok_i else ("⚠️" if ri.get("personaje_equivocado") else "❌")
                st.markdown(f"""<div style="background:linear-gradient(135deg,#12121f,#1a1a2e);border-radius:12px;
                    border-left:4px solid {color};padding:14px 18px;margin-bottom:10px;display:flex;flex-direction:column;gap:6px;">
                    <div style="font-family:'Bangers';color:{color};font-size:1.15rem;letter-spacing:1px;">{emoji} {personaje.upper()}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                        <div style="background:rgba(230,57,70,.08);border-radius:8px;padding:8px 12px;">
                            <span style="color:#e63946;font-family:Nunito;font-weight:700;font-size:.8rem;">🧑 VOS</span><br>
                            <span style="color:#fff;font-family:Nunito;font-size:.88rem;">{icon_u} {fb_u}</span>
                        </div>
                        <div style="background:rgba(51,153,255,.08);border-radius:8px;padding:8px 12px;">
                            <span style="color:#3399FF;font-family:Nunito;font-weight:700;font-size:.8rem;">🤖 IA</span><br>
                            <span style="color:#fff;font-family:Nunito;font-size:.88rem;">{icon_i} {fb_i}</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── Botón para ver imagen comparativa (punto 4) ──
            st.markdown("<br>", unsafe_allow_html=True)
            if "mostrar_comparacion" not in st.session_state:
                st.session_state.mostrar_comparacion = False

            col_cmp1, col_cmp2, col_cmp3 = st.columns([1, 2, 1])
            with col_cmp2:
                if st.button("🖼️ Ver comparación visual en el póster", use_container_width=True, type="secondary"):
                    st.session_state.mostrar_comparacion = not st.session_state.mostrar_comparacion

            if st.session_state.mostrar_comparacion:
                st.markdown("""<div style="background:#0d0d0d;border-radius:10px;padding:10px 16px;margin:10px 0;
                    font-family:Nunito;font-size:.85rem;color:#8b949e;text-align:center;">
                    🟡 Dorado = posición real &nbsp;·&nbsp; 🔴 Rojo = tu marcado &nbsp;·&nbsp; 🔵 Azul = detección IA
                </div>""", unsafe_allow_html=True)
                img_comparado = anotar_resultado_juego(img_pil, res_usuario, res_ia)
                # Limitar tamaño de visualización
                max_w = 900
                iw, ih = img_comparado.size
                if iw > max_w:
                    factor_vis = iw / max_w
                    img_comparado = img_comparado.resize((max_w, int(ih / factor_vis)), Image.LANCZOS)
                st.image(img_comparado, use_container_width=True)
                buf = io.BytesIO()
                img_pil_orig = Image.open(io.BytesIO(st.session_state.imagen_bytes)).convert("RGB")
                anotar_resultado_juego(img_pil_orig, res_usuario, res_ia).save(buf, format="PNG")
                col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                with col_dl2:
                    st.download_button("⬇️ Descargar comparación", data=buf.getvalue(),
                                       file_name="comparacion_waldo.png", mime="image/png",
                                       use_container_width=True)

    # 🤖 PÁGINA: RESOLVER CON IA (DROPZONE CENTRAL PREMIUM)
    elif st.session_state.page == "Resolver":
        if st.session_state.imagen_bytes is None:
            st.markdown("""
            <div style="text-align:center; padding: 20px 20px 10px;">
              <div style="font-size:5.5rem; line-height:1; margin-bottom:15px;">🤖</div>
              <h2 style="font-family:'Bangers',cursive; font-size:3.2rem; color:#3399FF; letter-spacing:2px; margin:0 0 10px;">ESCÁNER NEURONAL AUTOMÁTICO</h2>
              <p style="color:#c9d1d9; font-family:Nunito,sans-serif; font-size:1.15rem; margin-bottom:30px; max-width: 650px; margin-left: auto; margin-right: auto;">
                Cargá un archivo. La IA aplicará segmentación por parches <b>SAHI</b> para localizar los objetivos.
              </p>
            </div>""", unsafe_allow_html=True)

            col_izq, col_centro, col_der = st.columns([1, 2, 1])
            with col_centro:
                archivo_subido = st.file_uploader("📁 Arrastrá tu póster acá o hacé clic para buscar el archivo:", type=["jpg", "jpeg", "png", "webp"])
                if archivo_subido is not None:
                    st.session_state.imagen_key = f"resolver_{archivo_subido.name}_{archivo_subido.size}"
                    st.session_state.imagen_bytes = archivo_subido.read()
                    st.session_state.mostrar_solucion = False
                    st.rerun()
        else:
            st.markdown("<h2 style='font-family:Bangers; color:#3399FF; text-align:center; font-size: 2.5rem; margin-bottom: 25px;'>⚙️ Procesamiento de Inferencia</h2>", unsafe_allow_html=True)
            if st.button("⬅️ Cambiar imagen / Volver", use_container_width=True):
                st.session_state.imagen_bytes = None
                st.session_state.mostrar_solucion = False
                st.rerun()

            img_pil = Image.open(io.BytesIO(st.session_state.imagen_bytes)).convert("RGB")
            col_img, col_acciones = st.columns([2, 1])
            with col_img: st.image(img_pil, caption="Tu Póster Cargado", use_container_width=True)
            with col_acciones:
                st.markdown("""
                <div style="background: linear-gradient(145deg, #0f3460, #1a1a2e); padding: 25px; border-radius: 15px; border: 2px solid #3399FF; box-shadow: 0 10px 20px rgba(0,0,0,0.3);">
                    <h3 style="color: #3399FF; font-family: 'Bangers', cursive; font-size: 1.6rem; margin-top:0;">👁️ ANÁLISIS DE PARCHES</h3>
                    <p style="color: #c9d1d9; font-family: 'Nunito'; font-size: 0.95rem; line-height: 1.5;">
                        Se dividirá el plano general en mosaicos solapados aplicando un umbral estricto de confianza para descartar el ruido de fondo.
                    </p>
                </div><br>""", unsafe_allow_html=True)
                if st.button("🧠 Ejecutar Escáner IA", type="primary", use_container_width=True):
                    if modelo: st.session_state.mostrar_solucion = True
                    else: st.error("No se detectó best.pt en el directorio raíz.")

            if st.session_state.mostrar_solucion and modelo:
                st.divider()
                st.markdown("<h2 style='font-family:Bangers; color:#e63946; text-align:center; font-size:2.5rem;'>🎯 ENCONTRADOS POR LA IA</h2>", unsafe_allow_html=True)
                dets = obtener_detecciones(modelo, img_pil, st.session_state.imagen_key)

                cols = st.columns(5)
                for i, nombre in enumerate(CLASS_NAMES.values()):
                    cls_id = list(CLASS_NAMES.keys())[i]
                    n = sum(1 for d in dets if d["cls"]==cls_id)
                    col_c = CLASS_COLORS[nombre]
                    cols[i].markdown(f'<div style="text-align:center; padding:12px 5px; border-radius:12px; border:2px solid {col_c}; background: rgba(0,0,0,0.2);"><div style="font-size:1.8rem;">{CLASS_EMOJI[nombre]}</div><div style="font-family:Nunito; font-weight:800; color:{col_c}; font-size:.85rem; text-transform:uppercase;">{nombre}</div><div style="font-size:2.2rem; font-weight:900; color:#fff;">{n}</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if dets:
                    img_ann = anotar_imagen(img_pil, dets)
                    st.image(img_ann, caption="Plano Escaneado Anotado", use_container_width=True)
                    buf = io.BytesIO(); img_ann.save(buf, format="PNG")
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_down1, col_down2, col_down3 = st.columns([1,2,1])
                    with col_down2:
                        st.download_button(label="⬇️ Descargar Póster Resuelto en Alta Calidad", data=buf.getvalue(), file_name="Waldo_Solucion_IA.png", mime="image/png", use_container_width=True)

if __name__ == "__main__":
    main()
