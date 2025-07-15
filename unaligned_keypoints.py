import face_alignment                        # Deep learning-based facial landmark detector
from face_alignment import LandmarksType     # Enum for specifying type of landmarks (2D/3D)
import cv2                                   # OpenCV for video/image I/O and drawing
import numpy as np                           # NumPy for array and math operations
from tqdm import tqdm                        # Progress bar for loops
import os                                    # File/directory utilities
import json                                  # Saving and loading JSON files
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

# Assign diff color
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

# Initialize the face_alignment model (using 2D landmarks, False- don't flip horizontal, CPU)
fa = face_alignment.FaceAlignment(LandmarksType.TWO_D, flip_input=False, device='cpu')

#load input video
cap = cv2.VideoCapture("video1.mp4")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #get frame count
print("Total frames:", frame_count)                  #print count
print("Frame Rate: ", cv2.CAP_PROP_FRAME_COUNT)      #get rate

# output directories for saving annotated frames 
os.makedirs("output_unalignedFrames", exist_ok=True)

# List to store landmark coordinates for all frames, grouped by frame
all_grouped = []

# Empty Dictionary to get each feature's keypoint trajectories across frames
feature_trajectories = {k: [] for k in FACIAL_FEATURES}

#process each frame in videoo
for idx in tqdm(range(frame_count), desc="Processing frames"):
    ret, frame = cap.read()     #read next frame if not end (ret = TRUE)
    if not ret:
        break

    grouped_frame = {}          #For each frame store points for each feature
    preds = fa.get_landmarks(frame)     # get landmarks
    if preds is not None:
        keypoints = preds[0]

        for feature, indices in FACIAL_FEATURES.items():
            points = keypoints[indices].tolist()        #get keypoints for curr feature
            grouped_frame[feature] = points             #
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

raw_points = {}
diff_from_first = {}
diff_from_previous = {}

for feature, frames_points in feature_trajectories.items():
    frames_points = np.array(frames_points)  # (num_frames, num_points, 2)
    if len(frames_points.shape) != 3:
        continue  # skip if empty
    num_frames, num_points, _ = frames_points.shape

    # 1. Raw points: each point index gets a trajectory over all frames
    raw_points[feature] = {}
    for pt_idx in range(num_points):
        pt_traj = frames_points[:, pt_idx, :].tolist()
        raw_points[feature][str(pt_idx)] = pt_traj

    # 2. Difference from first frame
    diff_from_first[feature] = {}
    for pt_idx in range(num_points):
        ref = frames_points[0, pt_idx]
        diffs = []
        for frame_idx in range(num_frames):
            coord = frames_points[frame_idx, pt_idx]
            # handle missing points (nan)
            if np.any(np.isnan(coord)) or np.any(np.isnan(ref)):
                d = float('nan')
            else:
                d = float(np.linalg.norm(coord - ref))
            diffs.append([float(coord[0]), float(coord[1]), d])
        diff_from_first[feature][str(pt_idx)] = diffs

    # 3. Difference from previous frame
    diff_from_previous[feature] = {}
    for pt_idx in range(num_points):
        prev = frames_points[0, pt_idx]
        diffs = [[float(prev[0]), float(prev[1]), 0.0]]  # first frame, diff=0
        for frame_idx in range(1, num_frames):
            coord = frames_points[frame_idx, pt_idx]
            if np.any(np.isnan(coord)) or np.any(np.isnan(prev)):
                d = float('nan')
            else:
                d = float(np.linalg.norm(coord - prev))
            diffs.append([float(coord[0]), float(coord[1]), d])
            prev = coord
        diff_from_previous[feature][str(pt_idx)] = diffs

# Save to JSON
with open("raw_point_trajectories.json", "w") as f:
    json.dump(raw_points, f, indent=2)
with open("diff_from_first_frame.json", "w") as f:
    json.dump(diff_from_first, f, indent=2)
with open("diff_from_prev_frame.json", "w") as f:
    json.dump(diff_from_previous, f, indent=2)

def plot_feature_distances(data, output_folder, title_prefix):
    os.makedirs(output_folder, exist_ok=True)
    avg_distances_across_features = []

    for feature, points in data.items():
        plt.figure(figsize=(10, 5))
        point_dists = []
        for pt_idx, pt_values in points.items():
            distances = [d[2] for d in pt_values]
            plt.plot(distances, marker='o', markersize=3, label=f"pt {pt_idx}")  # line graph with circle markers
            point_dists.append(distances)
        plt.title(f"{title_prefix}: {feature}")
        plt.xlabel("Frame")
        plt.ylabel("Distance")
        plt.legend(loc='upper right', fontsize='x-small', ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{feature}.png"))
        plt.close()

        # Compute avg across points (ensure all have same frame length)
        min_len = min(len(d) for d in point_dists)
        trimmed = [d[:min_len] for d in point_dists]
        avg_dist = np.mean(trimmed, axis=0)
        avg_distances_across_features.append(avg_dist)

    # --- Overall average plot (avg across all features & points) --- #
    min_frames = min(arr.shape[0] for arr in avg_distances_across_features)
    trimmed = [arr[:min_frames] for arr in avg_distances_across_features]
    avg_across_all = np.mean(trimmed, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(avg_across_all, marker='o', markersize=3, label="Average across all features & points")
    plt.title(f"{title_prefix}: Average distance across all features & points")
    plt.xlabel("Frame")
    plt.ylabel("Average Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"overall_average.png"))
    plt.close()

# Load distance data from JSON
with open("diff_from_first_frame.json") as f:
    diff_first = json.load(f)
with open("diff_from_prev_frame.json") as f:
    diff_prev = json.load(f)

# Plot for diff from first frame
plot_feature_distances(
    diff_first,
    output_folder="graphs_diff_from_first_frame",
    title_prefix="Distance from First Frame"
)

# Plot for diff from previous frame
plot_feature_distances(
    diff_prev,
    output_folder="graphs_diff_from_prev_frame",
    title_prefix="Distance from Previous Frame"
)
