import face_alignment
from face_alignment import LandmarksType
import cv2
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt

# Facial feature index ranges
FACIAL_FEATURES = {
    "jawline": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68)),
}

FEATURE_COLORS = {
    "jawline": (255, 0, 0),
    "right_eyebrow": (0, 255, 0),
    "left_eyebrow": (0, 0, 255),
    "nose": (255, 255, 0),
    "right_eye": (255, 0, 255),
    "left_eye": (0, 255, 255),
    "outer_lip": (128, 0, 128),
    "inner_lip": (0, 128, 128),
}

fa = face_alignment.FaceAlignment(LandmarksType.TWO_D, flip_input=False, device='cpu')
cap = cv2.VideoCapture("video1.mp4")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", frame_count)

os.makedirs("output_unalignedFrames", exist_ok=True)
os.makedirs("graphs_unalignedKeypoints", exist_ok=True)

all_grouped = []
feature_trajectories = {k: [] for k in FACIAL_FEATURES}

for idx in tqdm(range(frame_count), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    grouped_frame = {}
    preds = fa.get_landmarks(frame)
    if preds is not None:
        keypoints = preds[0]

        for feature, indices in FACIAL_FEATURES.items():
            points = keypoints[indices].tolist()
            grouped_frame[feature] = points
            feature_trajectories[feature].append(points)

            for (x, y) in points:
                color = FEATURE_COLORS[feature]
                cv2.circle(frame, (int(x), int(y)), 2, color, -1)
    else:
        for feature, indices in FACIAL_FEATURES.items():
            points = [[np.nan, np.nan]] * len(indices)
            grouped_frame[feature] = points
            feature_trajectories[feature].append(points)

    all_grouped.append(grouped_frame)
    out_path = os.path.join("output_unalignedFrames", f"frame_{idx:04d}.jpg")
    cv2.imwrite(out_path, frame)

cap.release()

# Save JSON
with open("landmarks_grouped.json", "w") as f:
    json.dump(all_grouped, f, indent=2)

# Per-frame combined XY magnitude graph (1D scalar of position)
for feature, frames in feature_trajectories.items():
    mean_dists = []
    for pts in frames:
        pts = np.array(pts)
        if np.isnan(pts).any():
            mean_dists.append(np.nan)
        else:
            mean_pos = np.mean(pts, axis=0)  # [avg_x, avg_y]
            dist = np.linalg.norm(mean_pos)  # sqrt(x^2 + y^2)
            mean_dists.append(dist)

    plt.figure()
    plt.plot(mean_dists, marker='o', linestyle='-')
    plt.title(f"Movement of {feature} over frames (combined XY distance)")
    plt.xlabel("Frame")
    plt.ylabel("Mean XY magnitude (pixels)")
    plt.grid(True)
    plt.savefig(f"graphs_unalignedKeypoints/{feature}_movement.png")
    plt.close()

print("✅ Updated with single-line XY movement plots.")
