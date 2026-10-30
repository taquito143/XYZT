"""XYZT — desktop app.

Local server: opens a video, splits it into scenes with PySceneDetect, estimates
depth with Depth Anything V2 and builds the spacetime point cloud for each scene.
The interface opens in the browser and MP4s are saved to ./output.

Usage:  python app.py   [--port 8765] [--no-browser]
"""
import argparse, json, os, struct, subprocess, threading, time, uuid, webbrowser, traceback
import numpy as np
import cv2
from flask import Flask, Response, abort, jsonify, request, send_file

import depth

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
OUT = os.path.join(HERE, "output")
for d in (WORK, OUT, os.path.join(WORK, "videos"), os.path.join(WORK, "thumbs"), os.path.join(WORK, "clouds")):
    os.makedirs(d, exist_ok=True)

VFOV = np.deg2rad(64.0)
TANY = float(np.tan(VFOV / 2))

app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
videos = {}   # id -> {path, name, fps, frames, w, h, dur}
jobs = {}     # id -> {progress, msg, done, result, error}
gpu_lock = threading.Lock()


def new_job(fn, *args):
    jid = uuid.uuid4().hex[:10]
    jobs[jid] = {"progress": 0.0, "msg": "", "done": False, "result": None, "error": None}

    def report(msg, p=None):
        jobs[jid]["msg"] = msg
        if p is not None:
            jobs[jid]["progress"] = float(max(0, min(1, p)))

    def run():
        try:
            jobs[jid]["result"] = fn(report, *args)
            jobs[jid]["done"] = True
        except Exception as e:
            traceback.print_exc()
            jobs[jid]["error"] = str(e) or e.__class__.__name__

    threading.Thread(target=run, daemon=True).start()
    return jid


@app.get("/")
def index():
    return send_file(os.path.join(HERE, "static", "index.html"))


@app.get("/api/info")
def info():
    return jsonify(mode="desktop", device=depth.device_name(), output=OUT,
                   models={k: v[1] for k, v in depth.MODELS.items()})


@app.get("/api/job/<jid>")
def job(jid):
    j = jobs.get(jid) or abort(404)
    return jsonify(j)


def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("OpenCV cannot open this video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return dict(fps=fps, frames=n, w=w, h=h, dur=n / fps if fps else 0)


@app.post("/api/upload")
def upload():
    name = request.headers.get("x-filename", "video.mp4")
    from urllib.parse import unquote
    name = os.path.basename(unquote(name)) or "video.mp4"
    vid = uuid.uuid4().hex[:10]
    path = os.path.join(WORK, "videos", vid + os.path.splitext(name)[1].lower())
    with open(path, "wb") as f:
        while True:
            b = request.stream.read(1 << 20)
            if not b:
                break
            f.write(b)
    try:
        meta = probe(path)
    except Exception as e:
        return Response(str(e), 400)
    videos[vid] = dict(path=path, name=name, **meta)
    return jsonify(id=vid, name=name, dur=meta["dur"], w=meta["w"], h=meta["h"], fps=meta["fps"])


# ---------- scenes (PySceneDetect) ----------
def detect_scenes(report, vid, detector, threshold, min_len):
    from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector
    v = videos[vid]
    video = open_video(v["path"])
    fps = video.frame_rate
    min_frames = max(1, int(round(min_len * fps)))
    sm = SceneManager()
    if detector == "adaptive":
        sm.add_detector(AdaptiveDetector(adaptive_threshold=3.0, min_content_val=threshold * 0.55, min_scene_len=min_frames))
    else:
        sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_frames))
    dur = video.duration.get_seconds()
    pos = 0.0
    while pos < dur - 1e-3:
        pos = min(dur, pos + 5.0)
        sm.detect_scenes(video, end_time=pos)
        report(f"Looking for cuts · {pos:.0f}/{dur:.0f} s", 0.9 * pos / max(dur, 1e-6))
    lst = sm.get_scene_list(start_in_scene=True)
    scenes = [(a.get_seconds(), b.get_seconds()) for a, b in lst] or [(0.0, dur)]
    cap = cv2.VideoCapture(v["path"])
    out = []
    for i, (a, b) in enumerate(scenes):
        cap.set(cv2.CAP_PROP_POS_MSEC, (a + b) / 2 * 1000)
        ok, fr = cap.read()
        if ok:
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (320, max(2, int(320 * h / w))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(WORK, "thumbs", f"{vid}_{i}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append(dict(index=i, start=a, end=b))
        report("Thumbnails…", 0.9 + 0.1 * (i + 1) / len(scenes))
    cap.release()
    return {"scenes": out}


@app.post("/api/scenes")
def scenes():
    b = request.get_json()
    if b.get("id") not in videos:
        return Response("Unknown video; open it again.", 400)
    return jsonify(job=new_job(detect_scenes, b["id"], b.get("detector", "content"),
                               float(b.get("threshold", 27)), float(b.get("min_len", 1.0))))


@app.get("/api/thumb/<vid>/<int:i>")
def thumb(vid, i):
    p = os.path.join(WORK, "thumbs", f"{vid}_{i}.jpg")
    return send_file(p, mimetype="image/jpeg") if os.path.exists(p) else abort(404)


# ---------- point cloud for one scene ----------
def process_scene(report, vid, start, end, frames, wdef, maxp, model, grid):
    v = videos[vid]
    with gpu_lock:
        dm = depth.get_model(model, grid, report)
    fps, vw, vh = v["fps"], v["w"], v["h"]
    aspect = vw / vh
    wp, hp = int(wdef), max(2, int(round(wdef * vh / vw)))
    length = max(0.05, end - start)
    F = int(max(2, min(frames, int(length * fps))))
    keep_p = min(1.0, maxp * 1e6 / (F * wp * hp))
    hfov = 2 * np.arctan(TANY * aspect)
    rng = np.random.default_rng()
    jj, ii = np.mgrid[0:hp, 0:wp]
    cap = cv2.VideoCapture(v["path"])
    poses, chunks_pos, chunks_col, heads = [], [], [], []
    yaw = px = pz = 0.0
    prev = None
    t0 = time.time()
    for f in range(F):
        t = start + 0.04 + (length - 0.08) * f / max(1, F - 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (wp, hp), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(cv2.resize(rgb, (128, max(8, int(128 * vh / vw))), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
        if prev is not None:
            (dx, _dy), _r = cv2.phaseCorrelate(prev, g, cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F))
            yaw += dx / g.shape[1] * hfov
            px += -np.sin(yaw)
            pz += -np.cos(yaw)
        prev = g
        with gpu_lock:
            d = dm.infer(rgb)
        d = cv2.resize(d, (wp, hp), interpolation=cv2.INTER_LINEAR)
        lo, hi = np.percentile(d, 2), np.percentile(d, 98.5)
        dn = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
        mask = rng.random((hp, wp)) < keep_p if keep_p < 1 else np.ones((hp, wp), bool)
        n = int(mask.sum())
        u = ((ii[mask] + rng.random(n)) / wp * 2 - 1) * aspect
        vv = 1 - (jj[mask] + rng.random(n)) / hp * 2
        pos = np.stack([u, vv, dn[mask]], 1).astype(np.float32)
        chunks_pos.append(pos)
        chunks_col.append(small[mask].astype(np.uint8))
        heads.append(dict(n=n, x=float(px), z=float(pz), yaw=float(yaw)))
        el = time.time() - t0
        report(f"Depth ({depth.MODELS[model][1]}, {grid*14}px) · frame {f+1}/{F} · ~{int(el/(f+1)*(F-f-1))+1} s left", (f + 1) / F)
    cap.release()
    if not heads:
        raise ValueError("Could not read frames from this scene")
    for k, h in enumerate(heads):
        h["tn"] = k / max(1, len(heads) - 1)
    total = sum(h["n"] for h in heads)
    header = json.dumps(dict(frames=heads, total=total, aspect=aspect)).encode()
    key = uuid.uuid4().hex[:12]
    with open(os.path.join(WORK, "clouds", key + ".bin"), "wb") as f:
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        f.write(b"\0" * ((-(4 + len(header))) % 4))
        f.write(np.concatenate(chunks_pos).tobytes())
        f.write(np.concatenate(chunks_col).tobytes())
    return {"key": key, "frames": len(heads), "points": total}


@app.post("/api/process")
def process():
    b = request.get_json()
    if b.get("id") not in videos:
        return Response("Unknown video; open it again.", 400)
    model = b.get("model", "vits")
    if model not in depth.MODELS:
        return Response("Unknown model", 400)
    return jsonify(job=new_job(process_scene, b["id"], float(b["start"]), float(b["end"]), int(b.get("frames", 40)),
                               int(b.get("def", 160)), float(b.get("maxp", 3)), model, int(b.get("grid", 37))))


@app.get("/api/cloud/<key>")
def cloud(key):
    p = os.path.join(WORK, "clouds", os.path.basename(key) + ".bin")
    return send_file(p, mimetype="application/octet-stream") if os.path.exists(p) else abort(404)


# ---------- save and join ----------
def safe_name(n):
    n = os.path.basename(n or "video.mp4")
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in n)
    return keep or "video.mp4"


@app.post("/api/save")
def save():
    name = safe_name(request.args.get("name"))
    if not name.lower().endswith((".mp4", ".webm")):
        name += ".mp4"
    path = os.path.join(OUT, name)
    with open(path, "wb") as f:
        while True:
            b = request.stream.read(1 << 20)
            if not b:
                break
            f.write(b)
    return jsonify(path=path)


def concat(report, names):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    files = [os.path.join(OUT, safe_name(n)) for n in names if os.path.exists(os.path.join(OUT, safe_name(n)))]
    if len(files) < 2:
        raise ValueError("At least two saved videos are needed")
    lst = os.path.join(WORK, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in files:
            f.write("file '" + p.replace("'", "'\\''") + "'\n")
    base = os.path.splitext(os.path.basename(files[0]))[0].split("_scene")[0]
    out = os.path.join(OUT, f"{base}_all_{int(time.time())}.mp4")
    report("Joining videos…", 0.3)
    r = subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], capture_output=True, text=True)
    if r.returncode != 0:
        # different resolutions: re-encode
        report("Re-encoding to join…", 0.5)
        r = subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", out], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:])
    return {"path": out}


@app.post("/api/concat")
def concat_ep():
    return jsonify(job=new_job(concat, request.get_json().get("names", [])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/"
    print(f"\n  XYZT · desktop\n  Open {url}\n  Inference device: {depth.device_name()}\n  Output folder: {OUT}\n")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=a.port, threaded=True)
