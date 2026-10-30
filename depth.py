"""Depth Anything V2 (ONNX): download, resolution adaptation and inference.

The models exported by fabio-sim/Depth-Anything-ONNX have a fixed 518 px input
(a 37×37 grid of 14 px patches). To run at another resolution, the graph's shape
constants are rewritten and the positional embedding is interpolated, the same way
DINOv2 does internally.
"""
import os, threading, urllib.request
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
BASE_URL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/"
# name: (file in the release, label shown in the interface, size in bytes for validation)
MODELS = {
    "vits": ("depth_anything_v2_vits.onnx", "Small", 99_373_606),
    "vitb": ("depth_anything_v2_vitb.onnx", "Base", 389_230_102),
    "vitl": ("depth_anything_v2_vitl.onnx", "Large", 1_337_368_922),
}
# Native patch grid of the exported models: 518 px / 14 px per patch.
NATIVE = 37  # 518 / 14

# One model instance is cached; the lock keeps two jobs from loading at once.
_lock = threading.Lock()
_cache = {}


# Inference backends, best first. onnxruntime only lists those it was built with.
def providers():
    import onnxruntime as ort
    avail = ort.get_available_providers()
    pref = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in pref if p in avail] or avail


# Human-readable name of the device in use (shown in the interface badge).
def device_name():
    p = providers()[0]
    return {"CUDAExecutionProvider": "GPU NVIDIA", "DmlExecutionProvider": "GPU DirectML",
            "CoreMLExecutionProvider": "Apple CoreML", "CPUExecutionProvider": "CPU"}.get(p, p)


# Downloads a model once into models/, reporting progress. A partial file is never used.
def _download(name, report):
    fname, label, size = MODELS[name]
    os.makedirs(MODELS_DIR, exist_ok=True)
    dst = os.path.join(MODELS_DIR, fname)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return dst
    tmp = dst + ".part"
    with urllib.request.urlopen(BASE_URL + fname) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("content-length") or size)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            report(f"Downloading {label} model · {got/1e6:.0f}/{total/1e6:.0f} MB", got / total * 0.9)
    os.replace(tmp, dst)
    return dst


def _patch(src, dst, grid, report):
    """Rewrite the graph for a grid×grid patch layout (input of grid*14 px)."""
    import onnx, onnx.inliner
    from onnx import numpy_helper
    report("Adapting the model to the selected resolution…", 0.92)
    m = onnx.load(src)
    m = onnx.inliner.inline_local_functions(m)
    g = m.graph
    N = grid
    # Every shape constant that depends on the grid size, mapped to its value for the new grid:
    #   37 patches per side, 74 / 148 upsampled feature maps, 518 input pixels,
    #   1369 = 37² patch tokens, 1370 = 1369 + the class token.
    mp = {37: N, 74: 2 * N, 148: 4 * N, 518: 14 * N, 1369: N * N, 1370: N * N + 1}

    # Only small 1-D integer tensors are shape constants; weights are never touched.
    def remap(a):
        if a.dtype.kind == "i" and a.ndim == 1 and a.size <= 8 and any(int(v) in mp for v in a):
            return np.array([mp.get(int(v), int(v)) for v in a], dtype=a.dtype)
        return None

    # The positional embedding holds one vector per patch on a 37×37 grid.
    # It is resized with bicubic interpolation to N×N; the class-token vector is kept as is.
    for i, t in enumerate(g.initializer):
        a = numpy_helper.to_array(t)
        if t.name.endswith("pos_embed") and a.ndim == 3 and a.shape[1] == 1370:
            C = a.shape[2]
            cls, gridv = a[:, :1], a[0, 1:].reshape(37, 37, C)
            r = np.stack([cv2.resize(gridv[:, :, c], (N, N), interpolation=cv2.INTER_CUBIC) for c in range(C)], -1)
            na = np.concatenate([cls, r.reshape(1, N * N, C)], 1).astype(a.dtype)
            g.initializer[i].CopyFrom(numpy_helper.from_array(na, t.name))
            continue
        na = remap(a)
        if na is not None:
            g.initializer[i].CopyFrom(numpy_helper.from_array(na, t.name))
    # Shape constants can also live in Constant nodes instead of initializers.
    for node in g.node:
        if node.op_type != "Constant":
            continue
        for at in node.attribute:
            if at.name == "value" and at.t is not None:
                try:
                    a = numpy_helper.to_array(at.t)
                except Exception:
                    continue
                na = remap(a)
                if na is not None:
                    at.t.CopyFrom(numpy_helper.from_array(na, at.t.name))
    for d in (2, 3):
        g.input[0].type.tensor_type.shape.dim[d].dim_value = 14 * N
    for d in (1, 2):
        g.output[0].type.tensor_type.shape.dim[d].dim_value = 14 * N
    del g.value_info[:]
    onnx.save(m, dst)


# A loaded depth model at a fixed resolution.
class DepthModel:
    # Downloads, adapts (once, cached on disk as models/<name>_g<grid>.onnx) and loads the model.
    def __init__(self, name, grid, report=lambda *_: None):
        import onnxruntime as ort
        self.name, self.grid, self.size = name, grid, grid * 14
        src = _download(name, report)
        dst = os.path.join(MODELS_DIR, f"{name}_g{grid}.onnx")
        if not os.path.exists(dst):
            _patch(src, dst + ".tmp", grid, report)
            os.replace(dst + ".tmp", dst)
        report("Loading model…", 0.97)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            self.sess = ort.InferenceSession(dst, so, providers=providers())
        except Exception:
            self.sess = ort.InferenceSession(dst, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        # ImageNet normalization used during the model's training.
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def infer(self, rgb):
        """rgb: uint8 HxWx3 → relative disparity (size×size, larger = closer)."""
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        return self.sess.run(None, {self.inp: np.ascontiguousarray(x, np.float32)})[0][0]


# Returns the cached model for (name, grid), loading it if needed.
def get_model(name, grid, report=lambda *_: None):
    with _lock:
        key = (name, grid)
        if key not in _cache:
            _cache.clear()  # keep one model in memory at a time
            _cache[key] = DepthModel(name, grid, report)
        return _cache[key]
