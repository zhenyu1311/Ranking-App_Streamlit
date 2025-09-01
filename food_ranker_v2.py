# trueranker.py
# TrueRanker — Pairwise Item Ranker
# Winner-stays ladder + transitivity skip.
# NOW supports:
#   - Staging uploads: add single images one-by-one (mobile-friendly) and/or bulk images/ZIP
#   - "Start ranking" button to begin once you're ready
# Keeps: Local folder loader (desktop), pair-memory JSON, CSV/ZIP exports.
# Progress strips are not shown.
# Author : HEZHENYU GITHUB:zhenyu1311 1 SEP 2025

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
# Config / constants
# --------------------------
ALLOWED_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")  # Tip: add ".heic" with pillow-heif if needed

# --------------------------
# Data structures & helpers
# --------------------------
@dataclass
class PhotoItem:
    id: str          # unique id (filename only)
    name: str        # display name (filename)
    path: str        # absolute path or temp path

def pair_key(a: str, b: str) -> str:
    return f"{a}|{b}" if a < b else f"{b}|{a}"

def load_image(path: str, max_dim: int = 2000) -> Image.Image:
    """Load image and downscale the longest side to max_dim (keeps aspect)."""
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
# Local folder (desktop)
# --------------------------
def scan_folder_images(root_dir: str, recursive: bool = True) -> List[PhotoItem]:
    """Enumerate images from a local directory. IDs are filename-only."""
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
    # Sort for stable order
    paths.sort(key=lambda p: (os.path.dirname(p).lower(), os.path.basename(p).lower()))

    items_out: List[PhotoItem] = []
    seen_ids: set = set()
    for p in paths:
        fname = os.path.basename(p)
        if fname.lower().endswith(ALLOWED_EXTS):
            if fname in seen_ids:
                continue  # skip duplicate filename (ID collision)
            seen_ids.add(fname)
            items_out.append(PhotoItem(id=fname, name=fname, path=os.path.abspath(p)))
    return items_out

# --------------------------
# Upload staging (mobile/desktop)
# --------------------------
def ensure_workdir():
    """Create a per-session temp work directory once."""
    ss = st.session_state
    if "work_dir" not in ss or not ss["work_dir"] or not os.path.isdir(ss["work_dir"]):
        ss["work_dir"] = tempfile.mkdtemp(prefix="trueranker_")

def save_bytes_to_workdir(filename: str, data: bytes) -> str:
    """Write bytes to session work_dir using the original filename."""
    ensure_workdir()
    safe = filename.replace("\\", "_").replace("/", "_")
    out_path = os.path.join(st.session_state["work_dir"], safe)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path

def stage_add_single(upload) -> str:
    """Add one image to staging; returns message or raises."""
    if upload is None:
        return "No file selected."
    fname = os.path.basename(upload.name)
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ALLOWED_EXTS:
        return f"Unsupported file type: {fname}"
    # Duplicate filename check (ID = filename)
    if fname in st.session_state["staged_items"]:
        return f"Duplicate filename skipped: {fname}"
    data = upload.read()
    # Verify quickly
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        return f"Could not read image: {fname}"
    path = save_bytes_to_workdir(fname, data)
    st.session_state["staged_items"][fname] = PhotoItem(id=fname, name=fname, path=path)
    return f"Added: {fname}"

def stage_add_multiple(files) -> List[str]:
    msgs = []
    for uf in files or []:
        msg = stage_add_single(uf)
        if msg:
            msgs.append(msg)
    return msgs

def stage_add_zip(zip_file) -> List[str]:
    msgs = []
    if not zip_file:
        return msgs
    try:
        data = zip_file.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            names.sort(key=lambda n: (os.path.dirname(n).lower(), os.path.basename(n).lower()))
            for name in names:
                base = os.path.basename(name)
                if not base:
                    continue
                ext = os.path.splitext(base)[1].lower()
                if ext not in ALLOWED_EXTS:
                    continue
                if base in st.session_state["staged_items"]:
                    msgs.append(f"Duplicate filename skipped: {base}")
                    continue
                try:
                    payload = zf.read(name)
                    Image.open(io.BytesIO(payload)).verify()
                    path = save_bytes_to_workdir(base, payload)
                    st.session_state["staged_items"][base] = PhotoItem(id=base, name=base, path=path)
                    msgs.append(f"Added from ZIP: {base}")
                except Exception:
                    msgs.append(f"Could not read image in ZIP: {name}")
    except zipfile.BadZipFile:
        msgs.append("Uploaded ZIP is invalid or corrupted.")
    except Exception as e:
        msgs.append(f"Error reading ZIP: {e}")
    return msgs

# --------------------------
# Transitivity (skip redundant comparisons)
# --------------------------
def build_win_graph(pair_wins: Dict[str, str]) -> Dict[str, set]:
    """Adjacency: winner -> {losers} using direct decisions only."""
    g: Dict[str, set] = {}
    for k, winner in pair_wins.items():
        a, b = k.split("|", 1)
        loser = b if winner == a else a
        g.setdefault(winner, set()).add(loser)
        g.setdefault(loser, set())
    return g

def dominates_factory(graph: Dict[str, set]):
    """Return a memoized reachability checker over the given graph."""
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
    """Rebuild the transitivity dominance checker from current pair_wins."""
    g = build_win_graph(st.session_state["pair_wins"])
    st.session_state["dom_checker"] = dominates_factory(g)

# --------------------------
# Session state
# --------------------------
def ensure_state():
    ss = st.session_state
    ss.setdefault("mode", "Upload (mobile/desktop)")  # or "Local folder (desktop)"
    ss.setdefault("root_dir", "")
    ss.setdefault("recursive", True)

    ss.setdefault("staged_items", {})       # filename -> PhotoItem (uploads staging area)
    ss.setdefault("work_dir", "")           # temp folder for staged uploads

    ss.setdefault("items", [])              # active items after "Start ranking"
    ss.setdefault("id_to_item", {})         # id -> PhotoItem (active)
    ss.setdefault("pair_wins", {})          # "idA|idB" -> winner_id
    ss.setdefault("final_rank", [])         # champions in order
    ss.setdefault("pool", [])               # challengers THIS ROUND (excl contender)
    ss.setdefault("contender", None)        # current ladder winner
    ss.setdefault("current_pair", None)     # (contender_id, challenger_id)
    ss.setdefault("seed", 1337)
    ss.setdefault("dom_checker", None)

def reset_round():
    st.session_state["pool"] = []
    st.session_state["contender"] = None
    st.session_state["current_pair"] = None

def reset_all():
    # Best-effort: leave temp dir; many hosts clean /tmp periodically.
    st.session_state.clear()
    ensure_state()

# --------------------------
# Ladder logic (with transitivity)
# --------------------------
def start_new_round_if_needed():
    """If no active round, initialize contender + pool from remaining items."""
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
    """
    Winner-stays ladder:
      - Use direct memory or transitivity to auto-resolve if possible.
      - Otherwise set current_pair and return for user choice.
      - When pool empty, crown contender as champion and start next round.
    """
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

        # 1) Direct memory
        if k in ss["pair_wins"]:
            winner = ss["pair_wins"][k]
            if winner == ss["contender"]:
                ss["pool"].pop(0)  # challenger loses
            else:
                ss["contender"] = challenger  # challenger becomes new contender
                ss["pool"].pop(0)
            continue

        # 2) Transitivity
        if dom(ss["contender"], challenger):
            ss["pool"].pop(0)  # contender dominates challenger
            continue
        if dom(challenger, ss["contender"]):
            ss["contender"] = challenger  # challenger dominates contender
            ss["pool"].pop(0)
            continue

        # 3) Need a decision
        ss["current_pair"] = (ss["contender"], challenger)
        return

def record_user_choice(winner_id: str):
    """Record decision, update memory & transitivity, then continue."""
    ss = st.session_state
    if not ss["current_pair"]:
        return
    contender, challenger = ss["current_pair"]
    k = pair_key(contender, challenger)
    ss["pair_wins"][k] = winner_id
    rebuild_dom_checker()  # update transitivity

    if winner_id == contender:
        if ss["pool"] and ss["pool"][0] == challenger:
            ss["pool"].pop(0)
        else:
            ss["pool"] = [x for x in ss["pool"] if x != challenger]
    else:
        if ss["pool"] and ss["pool"][0] == challenger:
            ss["pool"].pop(0)
        else:
            ss["pool"] = [x for x in ss["pool"] if x != challenger]
        ss["contender"] = winner_id

    ss["current_pair"] = None
    advance_ladder_until_choice_needed()

# --------------------------
# UI
# --------------------------
st.set_page_config(page_title="TrueRanker — Pairwise Item Ranker", layout="wide")
ensure_state()

st.title("🔢 TrueRanker — Pairwise Item Ranker")

with st.expander("How it works", expanded=True):
    st.markdown("""
- Add items via **Upload** (mobile/desktop) — one by one or in bulk/ZIP — or from a **Local folder** (desktop).
- When ready, click **Start ranking** to begin.
- You’ll see two items at a time; pick the winner. The app **skips redundant matchups** using transitivity (if A>B and B>C, it infers A>C).
- Each finished round crowns the current contender as **Rank #1**, removes it, then finds #2, #3, …

**Notes**
- Runs on the server where Streamlit is hosted. Uploads are stored temporarily for this session.
- IDs use **filenames only** — avoid duplicate names.
""")

st.radio(
    "Select source",
    ["Upload (mobile/desktop)", "Local folder (desktop)"],
    key="mode",
    horizontal=True
)

if st.session_state["mode"] == "Upload (mobile/desktop)":
    st.subheader("Upload & Stage Items")

    # --- Single image (add 1-by-1) ---
    col_single, col_btn = st.columns([3, 1])
    with col_single:
        single_up = st.file_uploader("Add a single image", type=[e.lstrip(".") for e in ALLOWED_EXTS], key="single_up")
    with col_btn:
        if st.button("➕ Add image"):
            msg = stage_add_single(st.session_state.get("single_up"))
            if "Added:" in msg:
                st.success(msg)
            else:
                st.warning(msg)

    # --- Bulk images + ZIP (add to staging) ---
    up_images = st.file_uploader("Add multiple images (optional)", type=[e.lstrip(".") for e in ALLOWED_EXTS],
                                 accept_multiple_files=True, key="multi_up")
    up_zip = st.file_uploader("Or add a ZIP of images (optional)", type=["zip"], key="zip_up")

    bulk_col1, bulk_col2 = st.columns([1, 1])
    with bulk_col1:
        if st.button("➕ Add selected images"):
            msgs = stage_add_multiple(st.session_state.get("multi_up"))
            if msgs:
                for m in msgs:
                    (st.success if m.startswith("Added") else st.warning)(m)
            else:
                st.info("No images selected.")
    with bulk_col2:
        if st.button("➕ Add ZIP"):
            msgs = stage_add_zip(st.session_state.get("zip_up"))
            if msgs:
                for m in msgs:
                    (st.success if m.startswith("Added") else st.warning)(m)
            else:
                st.info("No ZIP selected.")

    # --- Staging preview & controls ---
    st.markdown("---")
    staged = st.session_state["staged_items"]
    st.markdown(f"**Staged items:** {len(staged)}")
    if staged:
        # Show filenames (simple text list to stay light/fast)
        names = sorted(staged.keys(), key=lambda x: x.lower())
        st.code("\n".join(names), language=None)

        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("🧹 Clear staging"):
                st.session_state["staged_items"] = {}
                st.info("Cleared staged items.")
        with c2:
            if st.button("✅ Start ranking (load staged items)"):
                # Commit staged items to active set
                items = [staged[n] for n in names]
                if len(items) < 2:
                    st.warning("Need at least 2 items to start ranking.")
                else:
                    st.session_state["items"] = items
                    st.session_state["id_to_item"] = {p.id: p for p in items}
                    st.session_state["pair_wins"] = {}
                    st.session_state["final_rank"] = []
                    st.session_state["dom_checker"] = None
                    reset_round()
                    # Kick off ladder and show first pair
                    start_new_round_if_needed()
                    advance_ladder_until_choice_needed()
                    st.success("Loaded staged items. Ranking started.")
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
    else:
        st.caption("Add images above, then click **Start ranking**.")

else:
    st.subheader("Local Folder (Desktop)")
    cA, cB = st.columns([3, 1])
    with cA:
        st.text_input("Folder path (on the server running Streamlit)", key="root_dir",
                      placeholder=r'Example: /app/images or C:\Users\you\Pictures')
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
                start_new_round_if_needed()
                advance_ladder_until_choice_needed()
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

# Pair-memory JSON load (optional)
mem_uploader = st.file_uploader("(Optional) Load Pair-Memory JSON", type=["json"], key="mem_json")
if mem_uploader:
    try:
        data = json.loads(mem_uploader.getvalue().decode("utf-8"))
        if isinstance(data, dict):
            st.session_state["pair_wins"].update(data)
            rebuild_dom_checker()
            st.success(f"Loaded pair memory with {len(data)} entries.")
    except Exception as e:
        st.warning(f"Could not load JSON: {e}")

# Reset
if st.button("🔄 Reset session"):
    reset_all()
    st.stop()

# Start/continue the ladder, auto-advance over known/transitive outcomes
start_new_round_if_needed()
advance_ladder_until_choice_needed()

# Summary
if st.session_state["items"]:
    total = len(st.session_state["items"])
    ranked = len(st.session_state["final_rank"])
    remain_round = (1 + len(st.session_state["pool"])) if st.session_state["contender"] else 0
    st.markdown(f"**Total:** {total} &nbsp; • &nbsp; **Ranked:** {ranked} &nbsp; • &nbsp; **Remaining this round:** {remain_round}")
else:
    st.info("Stage uploads above or load a local folder to begin.")

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
        if st.button("✅ Choose LEFT (Contender stays)"):
            record_user_choice(C.id)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with col2:
        st.image(imgD, caption=f"RIGHT (Challenger): {D.name}", use_container_width=False)
        if st.button("✅ Choose RIGHT (Challenger becomes new contender)"):
            record_user_choice(D.id)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
else:
    if st.session_state["items"] and len(st.session_state["final_rank"]) == len(st.session_state["items"]):
        st.success("🏁 All items ranked!")

# Current ranking preview (thumbnails)
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

st.markdown("---")

# --------------------------
# Exports
# --------------------------
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
        st.download_button("⬇️ Save Pair-Memory JSON", data=json_bytes,
                           file_name="pair_memory.json", mime="application/json")
    else:
        st.caption("Pair-memory JSON appears after you make at least one decision.")

with c_zip:
    if st.session_state["items"] and len(st.session_state["final_rank"]) == len(st.session_state["items"]):
        if st.button("⬇️ Build Ranked Items ZIP"):
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
            st.download_button("Save Ranked ZIP", data=zip_buf.getvalue(),
                               file_name="ranked_items.zip", mime="application/zip")
    else:
        st.caption("Ranked ZIP available after all items are fully ranked.")
