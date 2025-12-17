# Video-to-Spritesheet Pipeline

This repository provides a **two-stage pipeline** for converting video footage of people into **clean, animation-ready spritesheets** using **Robust Video Matting (RVM)**.

The pipeline is intentionally split so that:

- **Stage A** (slow, GPU-heavy) performs high-quality person matting once
- **Stage B** (fast, CPU-only) can be rerun many times to experiment with cropping, frame skipping, pixel-art quantisation, and spritesheet layout

---

## Overview

```
input/                   # Raw MP4 videos
intermediate/            # Stage A output (RGBA frames + metadata)
output_frames/           # Stage B output (spritesheets + frames)

stage_a_extract.py       # RVM-based person extraction (GPU recommended)
stage_b_postprocess.py   # Spritesheet generation + quantisation
requirements.txt
```

---

## Prerequisites

- Python **3.9+** (3.10 or 3.11 recommended)
- Git
- (Recommended) NVIDIA GPU with CUDA for Stage A

Stage B runs entirely on CPU.

---

## Clone the repository (with submodules)

This project depends on git submodules (e.g. **Robust Video Matting**), so you **must** initialise and update them.

Clone this repo then:

```bash
cd video2spritesheet
git submodule update --init --recursive
```

---

## Install Python dependencies

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
```

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Stage A — High-quality person extraction (RVM)

Stage A:
- Uses **Robust Video Matting**
- Produces full-resolution **RGBA frames**
- Computes global bounds and foot position
- Runs once per source video

### Input

Place MP4 videos in:

```
input/
```

### Run Stage A

```bash
python stage_a_extract.py
```

### Output

```
intermediate/<video_name>/
├── frames_rgba/
│   ├── frame_00000.png
│   └── ...
└── bounds.json
```

---

## Stage B — Post-processing & spritesheet generation

Stage B:
- Crops using Stage A metadata
- Normalises to fixed sprite size
- Performs motion-based frame skipping
- Supports target FPS + ping-pong safety
- Applies pixel-art quantisation
- Builds spritesheets

### Run Stage B

```bash
python stage_b_postprocess.py
```

### Output

```
output_frames/<video_name>/
├── frames/
├── <video_name>_spritesheet.png
└── <video_name>_meta.json
```

---

## Re-running Stage B

To experiment with visual styles or animation timing, edit
`stage_b_postprocess.py` and rerun **only Stage B**.

---

## Notes

- Videos should be fixed-camera and fully visible
- Even lighting improves segmentation quality
- Spritesheets are packed row-major, top-to-bottom

---

## License / Attribution

This project uses **Robust Video Matting** as a submodule.
Please ensure you comply with its license when redistributing or offering this as a service.

---

Happy animating ✨
