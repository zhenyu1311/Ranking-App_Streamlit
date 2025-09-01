# food_ranker_v2.py
# Local-folder ranking app (winner-stays ladder) with transitivity and exports.
# Optimized version (cached thumbnails, cached order) but progress strips are COMMENTED OUT.
#AUTHOR: HE ZHENYU. GITHUB:zhenyu1311 on 1 SEP 2025

import os
import io
import sys
import json
import glob
import random
import zipfile
import importlib
import subprocess
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# --------------------------
# Dependency bootstrap
# --------------------------
def ensure_deps(packages: List[str]):
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"[OK] {pkg} already installed")
        except ImportError:
            print(f"[MISSING] {pkg} not found, installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            importlib.invalidate_caches()
            importlib.import_module(pkg)
            print(f"[DONE] Installed {pkg}")

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
# Data structures & helpers
# --------------------------
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

@dataclass
class PhotoItem:
    id: str          # unique id (filename only)
    name: str        # display name (filename)
    path: str        # absolute path

def pair_key(a: str, b: str) -> str:
    return f"{a}|{b}" if a < b else f"{b}|{a}"

def scan_folder_images(root_dir: str, recursive: bool = True) -> List[PhotoItem]:
    """Enumerate images from a local directory (optionally recursive). IDs are filename-only."""
    items: List[PhotoItem] = []
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        return items

    patterns = []
    if recursive:
        for ext in ALLOWED_EXTS:
            patterns.append(os.path.join(root_dir, "**", f"*{ext}"))
    else:
        for ext in ALLOWED_EXTS:
            patterns.append(os.path.join(root_dir, f"*{ext}"))

    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))

    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort(key=lambda p: (os.path.dirname(p).lower(), os.path.basename(p).lower()))

    items_out: List[PhotoItem] = []
    for p in paths:
        fname = os.path.basename(p)
        items_out.append(PhotoItem(id=fname, name=fname, path=os.path.abspath(p)))
    return items_out

def load_image(path: str, max_dim: int = 2000) -> Image.Image:
    """Load image and downscale longest side to max_dim."""
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
    """Return a resized copy scaled for display."""
    w, h = img.size
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    if nw > max_width:
        nh = int(h * (max_width / float(w)))
        nw = max_width
    if nw == w and nh == h:
        return img
    return img.resize((nw, nh))

# --------------------------
# Transitivity (skip redundant comparisons)
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
        if a == b:
            return False
        stack = [a]
        seen = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for v in graph.get(u, ()):
                if v == b:
                    return True
                if v not in seen:
                    stack.append(v)
        return False
    return dominates

def rebuild_dom_checker():
    g = build_win_graph(st.session_state["pair_wins"])
    st.session_state["dom_checker"] = dominates_factory(g)

# --------------------------
# Session state
# --------------------------
def ensure_state():
    ss = st.session_state
    ss.setdefault("root_dir", "")
    ss.setdefault("recursive", True)
    ss.setdefault("items", [])
    ss.setdefault("id_to_item", {})
    ss.setdefault("pair_wins", {})
    ss.setdefault("final_rank", [])
    ss.setdefault("pool", [])
    ss.setdefault("contender", None)
    ss.setdefault("current_pair", None)
    ss.setdefault("seed", 1337)
    ss.setdefault("dom_checker", None)

def reset_round():
    st.session_state["pool"] = []
    st.session_state["contender"] = None
    st.session_state["current_pair"] = None

def reset_all():
    st.session_state.clear()
    ensure_state()

# --------------------------
# Ladder logic
# --------------------------
def start_new_round_if_needed():
    ss = st.session_state
    if ss["contender"] or ss["pool"]:
        return
    remaining_ids = [p.id for p in ss["items"] if p.id not in ss["final_rank"]]
    if not remaining_ids:
        return
    rng = random.Random(ss["seed"])
    rng.shuffle(remaining_ids)
    ss["contender"] = remaining_ids.pop(0)
    ss["pool"] = remaining_ids
    ss["current_pair"] = None

def advance_ladder_until_choice_needed():
    ss = st.session_state
    if ss["contender"] is None:
        return
    if ss["dom_checker"] is None:
        rebuild_dom_checker()
    dom = ss["dom_checker"]

    while True:
        if not ss["pool"]:
            champion = ss["contender"]
            ss["final_rank"].append(champion)
            reset_round()
            start_new_round_if_needed()
            if ss["contender"] is None:
                return
            else:
                continue

        challenger = ss["pool"][0]
        k = pair_key(ss["contender"], challenger)

        if k in ss["pair_wins"]:
            winner = ss["pair_wins"][k]
            if winner == ss["contender"]:
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

def record_user_choice(winner_id: str):
    ss = st.session_state
    if not ss["current_pair"]:
        return
    contender, challenger = ss["current_pair"]
    k = pair_key(contender, challenger)
    ss["pair_wins"][k] = winner_id
    rebuild_dom_checker()

    if winner_id == contender:
        ss["pool"] = [x for x in ss["pool"] if x != challenger]
    else:
        ss["pool"] = [x for x in ss["pool"] if x != challenger]
        ss["contender"] = winner_id

    ss["current_pair"] = None
    advance_ladder_until_choice_needed()

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="Pairwise Photo Ranker (Local + Transitivity)", layout="wide")
ensure_state()

st.title("🥇 Pairwise Photo Ranker — Local Folder (Ladder + Transitivity)")

with st.expander("Instructions", expanded=True):
    st.markdown("""
- Select a **local folder** of images (`.jpg/.jpeg/.png/.webp/.bmp`).
- Ladder rounds: **winner stays** vs next challenger.
- **Auto-skip** previously decided pairs **and** transitive outcomes (A>B, B>C ⇒ A>C).
- When a round ends, the contender becomes **Rank #1**, removed; next rounds find #2, #3, ...

- Pair memory stored as JSON (IDs = filenames).
- Exports: CSV, JSON, ZIP.
""")

cA, cB, cC = st.columns([3, 1, 1])
with cA:
    st.text_input("Folder path", key="root_dir", placeholder=r'Example: C:\Users\you\Pictures\Food')
with cB:
    st.checkbox("Include subfolders", key="recursive", value=True)

if st.button("📥 Load images from folder"):
    if not st.session_state["root_dir"]:
        st.error("Please enter a folder path.")
    elif not os.path.isdir(st.session_state["root_dir"]):
        st.error("That path is not a directory.")
    else:
        items = scan_folder_images(st.session_state["root_dir"], st.session_state["recursive"])
        if not items:
            st.error("No images found.")
        else:
            st.success(f"Loaded {len(items)} images.")
            st.session_state["items"] = items
            st.session_state["id_to_item"] = {p.id: p for p in items}
            st.session_state["pair_wins"] = {}
            st.session_state["final_rank"] = []
            st.session_state["dom_checker"] = None
            reset_round()

if st.button("🔄 Reset session"):
    reset_all()
    st.stop()

start_new_round_if_needed()
advance_ladder_until_choice_needed()

if st.session_state["items"]:
    total = len(st.session_state["items"])
    ranked = len(st.session_state["final_rank"])
    remain_round = (1 + len(st.session_state["pool"])) if st.session_state["contender"] else 0
    st.markdown(f"**Total:** {total} &nbsp; • &nbsp; **Ranked:** {ranked} &nbsp; • &nbsp; **Remaining this round:** {remain_round}")

    # --- COMMENTED OUT ---
    # if st.session_state["final_rank"]:
    #     render_progress_strip("Finished ranks:", st.session_state["final_rank"], st.session_state["id_to_item"])
    #
    # remaining_all = [pid for pid in st.session_state["id_to_item"].keys() if pid not in st.session_state["final_rank"]]
    # seen = seen_ids_from_memory(st.session_state["pair_wins"])
    # remaining_seen = [pid for pid in remaining_all if pid in seen]
    # if remaining_seen:
    #     ordered_remaining = topo_strip_order_cached(remaining_seen, st.session_state["pair_wins"], st.session_state["id_to_item"], st.session_state["contender"])
    #     render_progress_strip("Remaining contenders (seen so far):", ordered_remaining, st.session_state["id_to_item"])
    # ----------------------

# Choice UI
if st.session_state["items"] and st.session_state["current_pair"]:
    c_id, d_id = st.session_state["current_pair"]
    C = st.session_state["id_to_item"][c_id]
    D = st.session_state["id_to_item"][d_id]

    imgC = resize_for_display(load_image(C.path, max_dim=1600), scale=0.6, max_width=1000)
    imgD = resize_for_display(load_image(D.path, max_dim=1600), scale=0.6, max_width=1000)

    st.subheader("Choose the winner — Winner stays for the next comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.image(imgC, caption=f"LEFT (Contender): {C.name}", use_container_width=False)
        if st.button("✅ Choose LEFT"):
            record_user_choice(C.id)
            st.rerun()
    with col2:
        st.image(imgD, caption=f"RIGHT (Challenger): {D.name}", use_container_width=False)
        if st.button("✅ Choose RIGHT"):
            record_user_choice(D.id)
            st.rerun()
else:
    if st.session_state["items"] and len(st.session_state["final_rank"]) == len(st.session_state["items"]):
        st.success("🏁 All photos ranked!")

# Current ranking preview
if st.session_state["final_rank"]:
    st.subheader("Current Ranking (Top → Bottom)")
    cols = st.columns(5)
    for idx, pid in enumerate(st.session_state["final_rank"], start=1):
        p = st.session_state["id_to_item"][pid]
        try:
            thumb = resize_for_display(load_image(p.path, max_dim=1200), scale=0.45, max_width=500)
        except Exception:
            continue
        with cols[(idx - 1) % 5]:
            st.image(thumb, caption=f"#{idx}: {p.name}", use_container_width=False)

# Exports
st.markdown("---")
c_csv, c_json, c_zip = st.columns([1, 1, 2])

with c_csv:
    if st.session_state["items"] and st.session_state["final_rank"]:
        if st.button("⬇️ Build Ranking CSV"):
            lines = ["rank,filename,id,absolute_path"]
            for i, pid in enumerate(st.session_state["final_rank"], start=1):
                it = st.session_state["id_to_item"][pid]
                lines.append(f"{i},{it.name},{pid},{it.path}")
            csv_bytes = ("\n".join(lines)).encode("utf-8")
            st.download_button("Save CSV", data=csv_bytes, file_name="ranking.csv", mime="text/csv")

with c_json:
    if st.session_state["pair_wins"]:
        json_bytes = json.dumps(st.session_state["pair_wins"], indent=2).encode("utf-8")
        st.download_button("⬇️ Save Pair-Memory JSON", data=json_bytes, file_name="pair_memory.json", mime="application/json")

with c_zip:
    if st.session_state["items"] and len(st.session_state["final_rank"]) == len(st.session_state["items"]):
        if st.button("⬇️ Build Ranked Images ZIP"):
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                total = len(st.session_state["final_rank"])
                pad = max(2, len(str(total)))
                for rank_idx, pid in enumerate(st.session_state["final_rank"], start=1):
                    it = st.session_state["id_to_item"][pid]
                    base = os.path.basename(it.name).replace("\\", "_").replace("/", "_")
                    out_name = f"{str(rank_idx).zfill(pad)}_{base}"
                    try:
                        with open(it.path, "rb") as f:
                            zf.writestr(out_name, f.read())
                    except Exception:
                        continue
            st.download_button("Save Ranked ZIP", data=zip_buf.getvalue(), file_name="ranked_images.zip", mime="application/zip")

