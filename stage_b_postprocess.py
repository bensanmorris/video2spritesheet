import os
import cv2
import json
import math
import argparse
import numpy as np
from sklearn.cluster import KMeans

# ---------- ARGUMENTS ----------
parser = argparse.ArgumentParser(
    description="Stage B: Post-process RVM output into spritesheets"
)

parser.add_argument(
    "--input-dir",
    type=str,
    default="intermediate",
    help="Directory containing Stage A outputs"
)

parser.add_argument(
    "--output-dir",
    type=str,
    default="output_frames",
    help="Directory to write spritesheets and processed frames"
)

parser.add_argument(
    "--sprite-size",
    type=int,
    default=128,
    help="Final width/height of each sprite in pixels (square)"
)

parser.add_argument(
    "--frames-per-row",
    type=int,
    default=8,
    help="Number of frames per row in the spritesheet"
)

parser.add_argument(
    "--padding-ratio",
    type=float,
    default=0.12,
    help="Extra padding around detected bounds as a fraction of size"
)

parser.add_argument(
    "--pixel-scale",
    type=float,
    default=0.25,
    help="Downscale factor for pixel-art effect (e.g. 0.25)"
)

parser.add_argument(
    "--palette-colors",
    type=int,
    default=32,
    help="Maximum number of colours for pixel-art quantisation"
)

parser.add_argument(
    "--motion-threshold",
    type=float,
    default=0.01,
    help="Minimum alpha change required to keep a frame"
)

parser.add_argument(
    "--target-fps",
    type=int,
    default=12,
    help="Target animation FPS after frame skipping (0 disables FPS limiting)"
)

parser.add_argument(
    "--max-frames",
    type=int,
    default=64,
    help="Hard cap on number of frames per animation"
)

args = parser.parse_args()

# ---------- CONFIG ----------
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

SPRITE_SIZE = args.sprite_size
FRAMES_PER_ROW = args.frames_per_row
PADDING_RATIO = args.padding_ratio

PIXEL_SCALE = args.pixel_scale
PALETTE_COLORS = args.palette_colors

MOTION_THRESHOLD = args.motion_threshold
TARGET_FPS = args.target_fps
MAX_FRAMES = args.max_frames

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- pixel-art quantisation ----------
def pixel_art(rgba):
    h, w = rgba.shape[:2]

    small = cv2.resize(
        rgba,
        (int(w * PIXEL_SCALE), int(h * PIXEL_SCALE)),
        interpolation=cv2.INTER_NEAREST
    )

    alpha = small[:, :, 3]
    rgb = small[:, :, :3]

    fg = rgb[alpha > 0]
    if len(fg) > PALETTE_COLORS:
        kmeans = KMeans(n_clusters=PALETTE_COLORS, n_init=3)
        labels = kmeans.fit_predict(fg)
        palette = np.uint8(kmeans.cluster_centers_)
        rgb[alpha > 0] = palette[labels]

    small[:, :, :3] = rgb

    return cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

# ---------- process each animation ----------
for name in os.listdir(INPUT_DIR):
    src_dir = os.path.join(INPUT_DIR, name)
    frames_dir = os.path.join(src_dir, "frames_rgba")
    bounds_path = os.path.join(src_dir, "bounds.json")

    if not os.path.isdir(frames_dir) or not os.path.exists(bounds_path):
        continue

    with open(bounds_path) as f:
        meta = json.load(f)

    b = meta["global_bounds"]
    foot_y = meta["avg_foot_y"]

    out_dir = os.path.join(OUTPUT_DIR, name)
    out_frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(out_frames_dir, exist_ok=True)

    frame_files = sorted(os.listdir(frames_dir))
    loaded_frames = []
    alpha_list = []

    for fname in frame_files:
        frame = cv2.imread(
            os.path.join(frames_dir, fname),
            cv2.IMREAD_UNCHANGED
        )
        loaded_frames.append(frame)
        alpha_list.append(frame[:, :, 3].astype(np.float32) / 255.0)

    if not alpha_list:
        continue

    # ---------- motion-based + loop-friendly frame skipping ----------
    selected_indices = [0]
    prev_alpha = alpha_list[0]

    for i in range(1, len(alpha_list) - 1):
        alpha = alpha_list[i]
        diff = np.mean(np.abs(alpha - prev_alpha))
        if diff > MOTION_THRESHOLD:
            selected_indices.append(i)
            prev_alpha = alpha

    if len(alpha_list) > 1:
        selected_indices.append(len(alpha_list) - 1)

    # ---------- FPS-based downsampling ----------
    if TARGET_FPS > 0 and len(selected_indices) > TARGET_FPS:
        step = max(1, len(selected_indices) // TARGET_FPS)
        selected_indices = selected_indices[::step]

    # ---------- enforce MAX_FRAMES ----------
    if len(selected_indices) > MAX_FRAMES:
        step = len(selected_indices) / MAX_FRAMES
        selected_indices = [
            selected_indices[int(i * step)]
            for i in range(MAX_FRAMES)
        ]

    # ---------- crop & resize ----------
    min_x, min_y, max_x, max_y = b.values()
    w = max_x - min_x
    h = max_y - min_y

    min_x -= int(w * PADDING_RATIO)
    max_x += int(w * PADDING_RATIO)
    min_y -= int(h * PADDING_RATIO)
    max_y += int(h * PADDING_RATIO)

    size = max(max_x - min_x, max_y - min_y)
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    min_x = max(0, cx - size // 2)
    min_y = max(0, cy - size // 2)
    max_x = min(loaded_frames[0].shape[1], min_x + size)
    max_y = min(loaded_frames[0].shape[0], min_y + size)

    processed = []

    for i, idx in enumerate(selected_indices):
        frame = loaded_frames[idx]
        crop = frame[min_y:max_y, min_x:max_x]
        sprite = cv2.resize(
            crop,
            (SPRITE_SIZE, SPRITE_SIZE),
            interpolation=cv2.INTER_AREA
        )
        sprite = pixel_art(sprite)

        cv2.imwrite(
            os.path.join(out_frames_dir, f"frame_{i:05d}.png"),
            sprite
        )
        processed.append(sprite)

    # ---------- build spritesheet ----------
    rows = math.ceil(len(processed) / FRAMES_PER_ROW)
    sheet = np.zeros(
        (rows * SPRITE_SIZE, FRAMES_PER_ROW * SPRITE_SIZE, 4),
        dtype=np.uint8
    )

    for i, sprite in enumerate(processed):
        r, c = divmod(i, FRAMES_PER_ROW)
        sheet[
            r * SPRITE_SIZE:(r + 1) * SPRITE_SIZE,
            c * SPRITE_SIZE:(c + 1) * SPRITE_SIZE
        ] = sprite

    cv2.imwrite(
        os.path.join(out_dir, f"{name}_spritesheet.png"),
        sheet
    )

    # ---------- pivot calculation ----------
    pivot_y = int((foot_y - min_y) / size * SPRITE_SIZE)

    with open(os.path.join(out_dir, f"{name}_meta.json"), "w") as f:
        json.dump(
            {
                "sprite_size": SPRITE_SIZE,
                "frames": len(processed),
                "frames_per_row": FRAMES_PER_ROW,
                "pivot": {
                    "x": SPRITE_SIZE // 2,
                    "y": pivot_y
                }
            },
            f,
            indent=2
        )

    print(
        f"Processed {name}: "
        f"{len(processed)} frames, "
        f"pivot=({SPRITE_SIZE // 2}, {pivot_y})"
    )
