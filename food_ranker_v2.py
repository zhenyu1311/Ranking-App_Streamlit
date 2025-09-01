# trueranker.py
# TrueRanker — Pairwise Item Ranker
# Winner-stays ladder + transitivity skip.
# NOW supports:
#   - Staging uploads: add single images one-by-one (mobile-friendly) and/or bulk images/ZIP
#   - "Start ranking" button to begin once you're ready
# Keeps: Local folder loader (desktop), pair-memory JSON, CSV/ZIP exports.
# Progress strips are not shown.
# Author : HEZHENYU GITHUB:zhenyu1311 1 SEP 2025
# Shows choices in a horizontal scroll row (mobile-friendly).

import os
import io
import sys
import json
import glob
import random
import zipfile
import tempfile
import importlib
import subprocess
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Optional
import base64

# --------------------------
# Dependency bootstrap
# --------------------------
def ensure_deps(packages: List[str]):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            importlib.invalidate_caches()
            importlib.import_module(pkg)

try:
    import streamlit as st
except Exception:
    ensure_deps(["streamlit"])
    import streamlit as st

try:
    from PIL import Image
except Exception:
    ensure_deps(["pillow"])
    from PIL import Image

# --------------------------
# Config
# --------------------------
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

@dataclass
class PhotoItem:
    id: str
    name: str
    path: str

def pair_key(a: str, b: str) -> str:
    return f"{a}|{b}" if a < b else f"{b}|{a}"

def load_image(path: str, max_dim: int = 2000) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        if w >= h:
            nh = int(h * (max_dim / float(w)))
            img = img.resize((max_dim, nh))
        else:
            nw = int(w * (max_dim / float(h)))
            img = img.resize((nw, max_dim))
    return img

def resize_for_display(img: Image.Image, *, scale: float = 0.6, max_width: int = 1000) -> Image.Image:
    w, h = img.size
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    if nw > max_width:
        nh = int(h * (max_width / float(w)))
        nw = max_width
    return img.resize((nw, nh)) if (nw, nh) != (w, h) else img

# --------------------------
# Workdir
# --------------------------
def ensure_workdir():
    if "work_dir" not in st.session_state or not os.path.isdir(st.session_state["work_dir"]):
        st.session_state["work_dir"] = tempfile.mkdtemp(prefix="trueranker_")

def save_bytes(filename: str, data: bytes) -> str:
    ensure_workdir()
    safe = filename.replace("\\", "_").replace("/", "_")
    path = os.path.join(st.session_state["work_dir"], safe)
    with open(path, "wb") as f:
        f.write(data)
    return path

# --------------------------
# Upload ingestion
# --------------------------
def ingest_single(upload) -> Optional[PhotoItem]:
    if not upload:
        return None
    fname = os.path.basename(upload.name)
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ALLOWED_EXTS or fname in st.session_state["staged"]:
        return None
    data = upload.read()
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        return None
    path = save_bytes(fname, data)
    item = PhotoItem(id=fname, name=fname, path=path)
    st.session_state["staged"][fname] = item
    return item

def ingest_multiple(files) -> int:
    count = 0
    for f in files or []:
        if ingest_single(f):
            count += 1
    return count

def ingest_zip(upload) -> int:
    if not upload:
        return 0
    count = 0
    try:
        data = upload.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                base = os.path.basename(name)
                if not base:
                    continue
                ext = os.path.splitext(base)[1].lower()
                if ext not in ALLOWED_EXTS or base in st.session_state["staged"]:
                    continue
                payload = zf.read(name)
                try:
                    Image.open(io.BytesIO(payload)).verify()
                except Exception:
                    continue
                path = save_bytes(base, payload)
                st.session_state["staged"][base] = PhotoItem(id=base, name=base, path=path)
                count += 1
    except Exception:
        return 0
    return count

# --------------------------
# Transitivity
# --------------------------
def build_win_graph(pair_wins: Dict[str, str]) -> Dict[str, set]:
    g: Dict[str, set] = {}
    for k, winner in pair_wins.items():
        a, b = k.split("|", 1)
        loser = b if winner == a else a
        g.setdefault(winner, set()).add(loser)
        g.setdefault(loser, set())
    return g

def dominates_factory(graph: Dict[str, set]):
    @lru_cache(maxsize=None)
    def dominates(a: str, b: str) -> bool:
        if a == b: return False
        stack, seen = [a], set()
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u)
            for v in graph.get(u, ()):
                if v == b: return True
                stack.append(v)
        return False
    return dominates

def rebuild_dom_checker():
    st.session_state["dom_checker"] = dominates_factory(build_win_graph(st.session_state["pair_wins"]))

# --------------------------
# Session state
# --------------------------
def ensure_state():
    ss = st.session_state
    ss.setdefault("staged", {})
    ss.setdefault("items", [])
    ss.setdefault("id_to_item", {})
    ss.setdefault("pair_wins", {})
    ss.setdefault("final_rank", [])
    ss.setdefault("pool", [])
    ss.setdefault("contender", None)
    ss.setdefault("current_pair", None)
    ss.setdefault("dom_checker", None)

def reset_round():
    st.session_state["pool"] = []
    st.session_state["contender"] = None
    st.session_state["current_pair"] = None

# --------------------------
# Ladder
# --------------------------
def start_new_round_if_needed():
    ss = st.session_state
    if ss["contender"] or ss["pool"]: return
    remaining = [p.id for p in ss["items"] if p.id not in ss["final_rank"]]
    if not remaining: return
    random.shuffle(remaining)
    ss["contender"] = remaining.pop(0)
    ss["pool"] = remaining

def advance_until_choice():
    ss = st.session_state
    if not ss["contender"]: return
    if not ss["dom_checker"]: rebuild_dom_checker()
    dom = ss["dom_checker"]

    while True:
        if not ss["pool"]:
            ss["final_rank"].append(ss["contender"])
            reset_round()
            start_new_round_if_needed()
            if not ss["contender"]: return
            continue
        challenger = ss["pool"][0]
        k = pair_key(ss["contender"], challenger)
        if k in ss["pair_wins"]:
            if ss["pair_wins"][k] == ss["contender"]:
                ss["pool"].pop(0)
            else:
                ss["contender"] = challenger
                ss["pool"].pop(0)
            continue
        if dom(ss["contender"], challenger):
            ss["pool"].pop(0)
            continue
        if dom(challenger, ss["contender"]):
            ss["contender"] = challenger
            ss["pool"].pop(0)
            continue
        ss["current_pair"] = (ss["contender"], challenger)
        return

def record_choice(winner: str):
    ss = st.session_state
    if not ss["current_pair"]: return
    c, d = ss["current_pair"]
    ss["pair_wins"][pair_key(c, d)] = winner
    rebuild_dom_checker()
    if winner == c:
        ss["pool"] = [x for x in ss["pool"] if x != d]
    else:
        ss["pool"] = [x for x in ss["pool"] if x != d]
        ss["contender"] = winner
    ss["current_pair"] = None
    advance_until_choice()

# --------------------------
# Horizontal scroll display
# --------------------------
def show_side_by_side(imgs, captions, height=300):
    st.markdown(
        f"""
        <style>
        .scroll-container {{
            display: flex;
            overflow-x: auto;
        }}
        .scroll-container img {{
            max-height: {height}px;
            margin-right: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    html = '<div class="scroll-container">'
    for img, cap in zip(imgs, captions):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        html += f'<div><img src="data:image/png;base64,{b64}"><br><small>{cap}</small></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="TrueRanker", layout="wide")
ensure_state()

st.title("🔢 TrueRanker — Pairwise Item Ranker")

st.subheader("Upload Items")
col1, col2, col3 = st.columns(3)
with col1:
    single = st.file_uploader("Upload single image", type=[e.lstrip(".") for e in ALLOWED_EXTS], key="single")
    if st.button("➕ Add single"):
        if ingest_single(single): st.success("Upload successful.")
with col2:
    multi = st.file_uploader("Upload image file(s)", type=[e.lstrip(".") for e in ALLOWED_EXTS], accept_multiple_files=True, key="multi")
    if st.button("➕ Add files"):
        if ingest_multiple(multi): st.success("Upload successful.")
with col3:
    zipf = st.file_uploader("Upload ZIP file", type=["zip"], key="zip")
    if st.button("➕ Add ZIP"):
        if ingest_zip(zipf): st.success("Upload successful.")

# Start ranking prompt
if len(st.session_state["staged"]) >= 2 and not st.session_state["items"]:
    st.markdown("✅ Upload successful. Do you want to **start ranking**?")
    if st.button("Start ranking"):
        items = list(st.session_state["staged"].values())
        st.session_state["items"] = items
        st.session_state["id_to_item"] = {p.id: p for p in items}
        st.session_state["pair_wins"] = {}
        st.session_state["final_rank"] = []
        st.session_state["dom_checker"] = None
        reset_round()
        start_new_round_if_needed()
        advance_until_choice()
        st.rerun()

# Ranking UI
if st.session_state["items"] and st.session_state["current_pair"]:
    a, b = st.session_state["current_pair"]
    A, B = st.session_state["id_to_item"][a], st.session_state["id_to_item"][b]
    imgA = resize_for_display(load_image(A.path), scale=0.6)
    imgB = resize_for_display(load_image(B.path), scale=0.6)

    st.subheader("Choose the winner")
    show_side_by_side([imgA, imgB], [f"LEFT: {A.name}", f"RIGHT: {B.name}"])

    colA, colB = st.columns(2)
    with colA:
        if st.button("✅ Choose LEFT"):
            record_choice(A.id)
            st.rerun()
    with colB:
        if st.button("✅ Choose RIGHT"):
            record_choice(B.id)
            st.rerun()
elif st.session_state["items"] and len(st.session_state["final_rank"]) == len(st.session_state["items"]):
    st.success("🏁 All items ranked!")

# Ranking preview
if st.session_state["final_rank"]:
    st.subheader("Current Ranking")
    cols = st.columns(5)
    for i, pid in enumerate(st.session_state["final_rank"], 1):
        p = st.session_state["id_to_item"][pid]
        thumb = resize_for_display(load_image(p.path), scale=0.4)
        with cols[(i-1) % 5]:
            st.image(thumb, caption=f"#{i}: {p.name}")

