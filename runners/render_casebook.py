# runners/render_casebook.py
# -*- coding: utf-8 -*-
import os, json, argparse
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw


def safe_read_jsonl(p: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def load_dataset_index(dataset_jsonl: str) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for ex in safe_read_jsonl(dataset_jsonl):
        k = ex.get("id") or ex.get("image_id") or ex.get("study_id") or ex.get("uid") or ex.get("path")
        if k is None:
            continue
        idx[str(k)] = ex
    return idx


def find_image_path(case: Dict[str, Any], ex: Optional[Dict[str, Any]], image_root: Optional[str]) -> Optional[str]:
    for key in ["image_path", "image", "path", "img"]:
        p = case.get(key)
        if isinstance(p, str) and p and os.path.exists(p):
            return p

    if ex is not None:
        for key in ["image", "path", "img", "image_path"]:
            p = ex.get(key)
            if isinstance(p, str) and p:
                if os.path.exists(p):
                    return p
                if image_root:
                    cand = os.path.join(image_root, os.path.basename(p))
                    if os.path.exists(cand):
                        return cand

    img = case.get("image", "")
    if isinstance(img, str) and img:
        if os.path.exists(img):
            return img
        if image_root:
            cand = os.path.join(image_root, os.path.basename(img))
            if os.path.exists(cand):
                return cand
    return None


ACTION_KEYS = ("action", "a", "act", "token", "action_token", "decision")
ROI_KEYS = ("roi", "roi_xyxy", "kbcs_roi", "box", "gt_roi")
ADOPT_KEYS = ("adopted", "adopt", "did_adopt", "fused")


def _norm_action(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    u = s.upper().strip().strip("<>").strip()
    return u


def get_action(step: Dict[str, Any]) -> str:
    for k in ACTION_KEYS:
        v = step.get(k)
        if isinstance(v, str) and v.strip():
            return _norm_action(v)
    return ""


def get_roi_xyxy(step: Dict[str, Any]) -> Optional[List[int]]:
    for k in ROI_KEYS:
        v = step.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            try:
                x1, y1, x2, y2 = map(float, v)
                return [int(x1), int(y1), int(x2), int(y2)]
            except Exception:
                pass

    v = step.get("roi_xywh")
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            x, y, w, h = map(float, v)
            return [int(x), int(y), int(x + w), int(y + h)]
        except Exception:
            pass
    return None


def get_float(step: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for k in keys:
        v = step.get(k)
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None


def get_adopt_flag(step: Dict[str, Any]) -> Optional[bool]:
    for k in ADOPT_KEYS:
        if k not in step:
            continue
        v = step.get(k)
        try:
            return bool(int(v))
        except Exception:
            return bool(v)
    return None


def infer_adopted_by_delta(step: Dict[str, Any], eps: float = 1e-3) -> bool:
    pb = get_float(step, ("p_before", "pb", "p0"))
    pa = get_float(step, ("p_after", "pa", "p1"))
    if pb is None or pa is None:
        return False
    try:
        return abs(float(pa) - float(pb)) > eps
    except Exception:
        return False


def is_empty_step(step: Dict[str, Any]) -> bool:
    if not isinstance(step, dict) or not step:
        return True
    for k in ACTION_KEYS + ROI_KEYS + ("p_before", "p_after", "q", "margin", "tool_time_ms", "tool_ms"):
        if k in step:
            return False
    return True


def union_boxes(boxes: List[List[int]]) -> Optional[List[int]]:
    if not boxes:
        return None
    xs1 = [b[0] for b in boxes]
    ys1 = [b[1] for b in boxes]
    xs2 = [b[2] for b in boxes]
    ys2 = [b[3] for b in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def clamp_box(b: List[int], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def dilate_box(b: List[int], dil: int) -> List[int]:
    x1, y1, x2, y2 = b
    return [x1 - dil, y1 - dil, x2 + dil, y2 + dil]


def draw_rois(img_path: str, traj: List[Dict[str, Any]], out_png: str):
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    k = 0
    for step in traj:
        if not isinstance(step, dict):
            continue
        a = get_action(step)
        if a != "CHECK":
            continue
        roi = get_roi_xyxy(step)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        k += 1
        draw.rectangle([x1, y1, x2, y2], width=4)
        draw.text((x1 + 3, y1 + 3), f"CHECK#{k}", fill=(255, 255, 255))

    im.save(out_png)


def draw_masked(img_path: str, mask_box: List[int], out_png: str, dilate: int = 12) -> List[int]:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    b = dilate_box(mask_box, dilate)
    b = clamp_box(b, w, h)
    draw = ImageDraw.Draw(im)
    draw.rectangle(b, fill=(0, 0, 0))
    im.save(out_png)
    return b


def traj_to_table(traj: List[Dict[str, Any]], eps: float = 1e-3):
    cols = ["action", "p_before", "q", "adopted", "margin", "roi", "p_after", "tool_time_ms", "decision"]
    header = ["t"] + cols
    rows = []
    t = 0

    for st in traj:
        if is_empty_step(st):
            continue
        t += 1
        row = [str(t)]
        for c in cols:
            if c == "action":
                v = get_action(st)
            elif c == "adopted":
                flag = get_adopt_flag(st)
                if flag is None:
                    flag = infer_adopted_by_delta(st, eps=eps)
                v = int(bool(flag))
            elif c == "roi":
                v = get_roi_xyxy(st) or ""
            elif c == "tool_time_ms":
                v = st.get("tool_time_ms", st.get("tool_ms", ""))
            else:
                v = st.get(c, "")

            if isinstance(v, float):
                row.append(f"{v:.3f}")
            else:
                row.append(str(v))
        rows.append(row)

    return header, rows


def score_case(case: Dict[str, Any], eps: float = 1e-3) -> Tuple[int, int, int, float]:
    traj = case.get("traj", [])
    if not isinstance(traj, list):
        return (0, 0, 0, 0.0)
    clean = [st for st in traj if isinstance(st, dict) and not is_empty_step(st)]
    n_steps = len(clean)

    n_check = 0
    n_adopt = 0
    max_dp = 0.0
    for st in clean:
        if get_action(st) == "CHECK" and get_roi_xyxy(st) is not None:
            n_check += 1
        flag = get_adopt_flag(st)
        if flag is None:
            flag = infer_adopted_by_delta(st, eps=eps)
        if flag:
            n_adopt += 1
        pb = get_float(st, ("p_before", "pb", "p0"))
        pa = get_float(st, ("p_after", "pa", "p1"))
        if pb is not None and pa is not None:
            try:
                max_dp = max(max_dp, abs(float(pa) - float(pb)))
            except Exception:
                pass
    return (n_steps, n_check, n_adopt, float(max_dp))


def pick_one_case(
    cases: List[Dict[str, Any]],
    min_steps: int,
    require_adopt: bool,
    eps: float,
    pick_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    # if pick_id specified, return that case
    if pick_id:
        for c in cases:
            cid = c.get("id") or c.get("uid") or c.get("image_id") or c.get("study_id")
            if cid is not None and str(cid) == str(pick_id):
                return c
        return None

    cand = []
    for c in cases:
        traj = c.get("traj", [])
        if not isinstance(traj, list):
            continue
        clean = [st for st in traj if isinstance(st, dict) and not is_empty_step(st)]
        if len(clean) < min_steps:
            continue
        if not any(get_action(st) == "CHECK" for st in clean):
            continue
        if require_adopt:
            ok = False
            for st in clean:
                flag = get_adopt_flag(st)
                if flag is None:
                    flag = infer_adopted_by_delta(st, eps=eps)
                if flag:
                    ok = True
                    break
            if not ok:
                continue
        cand.append(c)

    if not cand:
        return None

    cand.sort(key=lambda x: score_case(x, eps=eps), reverse=True)
    return cand[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="artifacts/debug/rl_traj_cases.jsonl")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", default="artifacts/casebook_one")
    ap.add_argument("--image_root", default=None)

    ap.add_argument("--min_steps", type=int, default=2)
    ap.add_argument("--require_adopt", action="store_true")
    ap.add_argument("--pick_id", type=str, default=None, help="render this exact case id (from traj jsonl)")

    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--mask_dilate", type=int, default=12)
    ap.add_argument("--eps", type=float, default=1e-3)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds_idx = load_dataset_index(args.dataset)
    cases = safe_read_jsonl(args.cases)

    picked = pick_one_case(
        cases,
        min_steps=args.min_steps,
        require_adopt=args.require_adopt,
        eps=float(args.eps),
        pick_id=args.pick_id,
    )
    if picked is None:
        raise SystemExit("[ERR] No suitable case found (or pick_id not found).")

    concept = picked.get("concept", "")
    traj = picked.get("traj", [])
    key = picked.get("id") or picked.get("image_id") or picked.get("study_id") or picked.get("uid")
    ex = ds_idx.get(str(key), None) if key is not None else None

    img_path = find_image_path(picked, ex, args.image_root)
    if img_path is None:
        raise SystemExit("[ERR] Could not resolve image path. Try --image_root or ensure dataset has absolute paths.")

    # ROI overlay
    png_rois = os.path.join(args.out_dir, "case_001_rois.png")
    draw_rois(img_path, traj, png_rois)

    # Mask
    mask_info = ""
    png_mask = None
    if args.mask:
        rois_all = []
        rois_adopt = []
        for st in traj:
            if not isinstance(st, dict) or is_empty_step(st):
                continue
            if get_action(st) != "CHECK":
                continue
            roi = get_roi_xyxy(st)
            if roi is None:
                continue
            rois_all.append(roi)
            flag = get_adopt_flag(st)
            if flag is None:
                flag = infer_adopted_by_delta(st, eps=float(args.eps))
            if flag:
                rois_adopt.append(roi)

        use_rois = rois_adopt if rois_adopt else rois_all
        u = union_boxes(use_rois) if use_rois else None
        if u is not None:
            png_mask = os.path.join(args.out_dir, "case_001_mask.png")
            b = draw_masked(img_path, u, png_mask, dilate=int(args.mask_dilate))
            mask_info = f"Mask box (dilated): {b} (base union: {u})"
        else:
            mask_info = "Mask requested but no ROI found in traj."

    header, rows = traj_to_table(traj, eps=float(args.eps))

    html = []
    html.append("<html><head><meta charset='utf-8'><style>")
    html.append("body{font-family:Arial, sans-serif;} .case{margin:20px;padding:20px;border:1px solid #ddd;}")
    html.append("table{border-collapse:collapse;} td,th{border:1px solid #aaa;padding:4px 6px;font-size:12px;}")
    html.append(".row{display:flex;gap:12px;flex-wrap:wrap;} .col{flex:1;min-width:320px;}")
    html.append("</style></head><body>")
    html.append("<h2>Casebook (single case)</h2>")

    html.append("<div class='case'>")
    html.append(f"<h3>concept = {concept}</h3>")
    if key is not None:
        html.append(f"<p><b>id:</b> {key}</p>")
    html.append(f"<p><b>image:</b> {os.path.basename(img_path)}</p>")

    html.append("<div class='row'>")
    html.append("<div class='col'><h4>ROI Overlay</h4>")
    html.append("<img src='case_001_rois.png' style='max-width:900px;width:100%;'/></div>")
    if png_mask is not None:
        html.append("<div class='col'><h4>Masked ROI (for intervention)</h4>")
        html.append("<img src='case_001_mask.png' style='max-width:900px;width:100%;'/>")
        if mask_info:
            html.append(f"<p><b>{mask_info}</b></p>")
        html.append("</div>")
    html.append("</div>")

    html.append("<h4>Trajectory</h4>")
    html.append("<table><tr>" + "".join([f"<th>{h}</th>" for h in header]) + "</tr>")
    for r in rows:
        html.append("<tr>" + "".join([f"<td>{x}</td>" for x in r]) + "</tr>")
    html.append("</table></div></body></html>")

    out_html = os.path.join(args.out_dir, "casebook.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print("[ok] wrote:", out_html)
    print("[ok] used image:", img_path)


if __name__ == "__main__":
    main()
