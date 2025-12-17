import os
import cv2
import json
import torch
import numpy as np
import sys

RVM_PATH = os.path.join(os.path.dirname(__file__), "RobustVideoMatting")
sys.path.insert(0, RVM_PATH)
from model import MattingNetwork

INPUT_DIR = "input"
OUTPUT_DIR = "intermediate"
ALPHA_THRESHOLD = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MattingNetwork("resnet50")
model.load_state_dict(torch.load("rvm_resnet50.pth", map_location=device))
model = model.to(device).eval()

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".mp4"):
        continue

    name = os.path.splitext(filename)[0]
    print(f"\nExtracting {name}")

    out_dir = os.path.join(OUTPUT_DIR, name)
    frames_dir = os.path.join(out_dir, "frames_rgba")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(os.path.join(INPUT_DIR, filename))

    bounds = []
    foot_y = []
    rec = [None, None, None, None]

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            _, pha, *rec = model(tensor, *rec)

        alpha = pha[0, 0].cpu().numpy()

        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (alpha * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(frames_dir, f"frame_{frame_index:05d}.png"),
            rgba
        )

        ys, xs = np.where(alpha > ALPHA_THRESHOLD)
        if len(xs) > 0:
            bounds.append((xs.min(), ys.min(), xs.max(), ys.max()))
            foot_y.append(ys.max())

        frame_index += 1

    cap.release()

    if not bounds:
        print("  No foreground detected, skipping")
        continue

    meta = {
        "global_bounds": {
            "min_x": int(min(b[0] for b in bounds)),
            "min_y": int(min(b[1] for b in bounds)),
            "max_x": int(max(b[2] for b in bounds)),
            "max_y": int(max(b[3] for b in bounds))
        },
        "avg_foot_y": int(np.mean(foot_y))
    }

    with open(os.path.join(out_dir, "bounds.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Frames written: {frame_index}")
