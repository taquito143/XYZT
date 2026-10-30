# XYZT

Turn a video into a **spacetime point cloud (x, y, z, t)**. Every frame is projected into 3D using its estimated depth and its moment in time, and all frames stay visible at once. A virtual camera then flies through that volume and the result is exported as MP4.

- **Scenes:** the video is split into scenes with [PySceneDetect](https://www.scenedetect.com/); pick which ones to convert.
- **Depth:** [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (Small, Base or Large) at 308–770 px.
- **Camera:** free orbit, automatic flight along the walk, or your own keyframed path with smooth curves.
- **Guides:** XYZ axes, ground grid, camera path and a split view (3D · top · side). None of it is rendered into the video.
- **Export:** one MP4 per scene, rendered frame by frame, plus an option to join them into a single file.

Everything runs locally: a small Python server does the heavy work and the interface opens in your browser.

## Requirements

- Python 3.10, 3.11 or 3.12
- Chrome or Edge (MP4 export uses WebCodecs)

## Install and run

- **Windows:** double-click `run.bat`
- **macOS / Linux:** `./run.sh`

The first run creates a local environment (`.venv`) and installs the dependencies. It then opens `http://127.0.0.1:8765/`. If a dependency goes missing, the script reinstalls it on the next run.

Manual alternative:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python app.py                        # Windows: .venv\Scripts\python app.py
```

Options: `--port 8765` and `--no-browser`.

## Workflow

1. **Open video:** the file is copied into the app.
2. **Scenes:** choose a detector (*Content* for hard cuts, *Adaptive* for moving-camera footage), threshold and minimum scene length.
3. **Process:** set frames per scene, point detail, max particles, depth model and depth resolution.
4. **Style:** trail length, turn following, background cut, stroke size, ink, wash, fog, wave, drift and dust.
5. **Camera:** frame a shot and press ◆ to add a keyframe; edit keyframe times, preview the path, toggle guides and split view.
6. **Export:** resolution (480p–4K), format (9:16, 1:1, 16:9), fps, camera motion and duration. Then **Process and export** in the Scenes tab renders one MP4 per selected scene into `output/`.

## Depth models

Downloaded automatically the first time you pick them:

| Model | Download | Notes |
|---|---|---|
| Small | 99 MB | fast, good for testing |
| Base | 389 MB | more detail |
| Large | 1.3 GB | most detail; a GPU is recommended |

**Depth resolution** sets the size the depth is estimated at. 518 px is the model's native size; other sizes are produced by rewriting the ONNX graph's shape constants and interpolating the positional embedding.

## GPU acceleration (optional)

By default inference runs on the CPU. To use your graphics card, swap onnxruntime inside the environment:

- NVIDIA (CUDA): `pip uninstall -y onnxruntime && pip install onnxruntime-gpu`
- Windows, any GPU (DirectML): `pip uninstall -y onnxruntime && pip install onnxruntime-directml`
- Apple Silicon: CoreML is used automatically.

The badge in the top-left corner shows which device is in use.

## How it works

1. **Scene cuts:** PySceneDetect compares consecutive frames (hue, saturation and brightness difference) and marks a cut above the threshold.
2. **Depth:** each sampled frame goes through Depth Anything V2 and is normalized per frame with robust percentiles.
3. **Path:** the turn between frames is estimated with phase correlation; each frame is placed one step further along that heading.
4. **Point cloud:** each pixel becomes a point (x, y, z) with its color and time `t`. The viewer (three.js) draws every frame at once and offsets it by the time trail.
5. **Video:** frames are rendered offline, encoded to H.264 with WebCodecs, muxed with mp4-muxer and saved to `output/`. Joining uses ffmpeg (bundled through imageio-ffmpeg).

## Folders

- `output/`: rendered videos
- `work/`: uploaded videos, thumbnails and temporary point clouds (safe to delete)
- `models/`: downloaded depth models

## Credits

- [PySceneDetect](https://www.scenedetect.com/)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and the ONNX exports by [fabio-sim/Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX)
- [three.js](https://threejs.org/), [mp4-muxer](https://github.com/Vanilagy/mp4-muxer), [ONNX Runtime](https://onnxruntime.ai/), [OpenCV](https://opencv.org/)
