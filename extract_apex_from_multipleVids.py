import face_alignment                       # deep learning facial landmarks
from face_alignment import LandmarksType
import cv2                                  # image and video processing
import numpy as np
from tqdm import tqdm                       # progress bar for loops
import os                                   # folder and path manipulation
import matplotlib.pyplot as plt
import subprocess                           #you use it to call ffmpeg

# Import conversion helpers from your separate script

# Facial Feature Ranges & Colors 
# Pre-defining which indices in the 68-point landmark array belong to each facial part.
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

# Extracts facial keypoints and saves frames with color-coded features
def process_video_keypoints(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {frame_count}, Frame Rate: {frame_rate}")
    os.makedirs(frames_dir, exist_ok=True)

    # Initialize face alignment model for 2D landmark detection
    fa = face_alignment.FaceAlignment(LandmarksType.TWO_D, flip_input=False, device='cpu')

    # A list: organized by frame first then feature
    all_grouped = []
    # A dictionary: organized by feature first then frame
    feature_trajectories = {k: [] for k in FACIAL_FEATURES}

    # Iterate through frame idx (display progress bar and speed)
    for idx in tqdm(range(frame_count), desc=f"Frames for {os.path.basename(video_path)}"):
        ret, frame = cap.read()         # read frame
        if not ret:
            break

        # Creates empty dictionary to store lists of keypoints for each feature for this frame
        grouped_frame = {}
        preds = fa.get_landmarks(frame)
        if preds is not None:                  # using first face's landmark
            keypoints = preds[0]               # numpy array w (68,2) shape
            for feature, indices in FACIAL_FEATURES.items():
                points = keypoints[indices].tolist()                    # get coords and convert to list [,]
                grouped_frame[feature] = points                         # save points for this feature
                feature_trajectories[feature].append(points)
                for (x, y) in points:
                    color = FEATURE_COLORS[feature]
                    cv2.circle(frame, (int(x), int(y)), 2, color, -1)
        else:
            # No face detected, creates a list of [nan, nan] (missing value)
            for feature, indices in FACIAL_FEATURES.items():
                points = [[np.nan, np.nan]] * len(indices)
                grouped_frame[feature] = points
                feature_trajectories[feature].append(points)

        # Saves this frame’s feature points to big list
        all_grouped.append(grouped_frame)
        # Create annotated frames
        out_path = os.path.join(frames_dir, f"frame_{idx:04d}.jpg")
        cv2.imwrite(out_path, frame)

    cap.release()
    return all_grouped, feature_trajectories, frame_rate

# Plots graphs for each feature, returns APEX INFO
def plot_feature_distances(diff_from_first, graphs_dir, video_title, frame_rate):
    os.makedirs(graphs_dir, exist_ok=True)
    avg_distances_across_features = []  # empty list to store avg distance

    for feature, points in diff_from_first.items():
        plt.figure(figsize=(10, 5))
        # Store just the distances (not the x/y)
        point_dists = []
        # Plot every keypoint for this feature
        for pt_idx, pt_values in points.items():
            distances = [d[2] for d in pt_values]
            plt.plot(distances, marker='o', markersize=3, label=f"pt {pt_idx}")
            point_dists.append(distances)
        plt.title(f"Distance from First Frame: {feature}")
        plt.xlabel("Frame")
        plt.ylabel("Distance")
        plt.legend(loc='upper right', fontsize='x-small', ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"{feature}.png"))
        plt.close()

        # Average distance across all keypoints for this feature
        min_len = min(len(d) for d in point_dists)
        trimmed = [d[:min_len] for d in point_dists]
        avg_dist = np.mean(trimmed, axis=0)
        avg_distances_across_features.append(avg_dist)

    # After all features: compute overall average across features
    min_frames = min(arr.shape[0] for arr in avg_distances_across_features)
    trimmed = [arr[:min_frames] for arr in avg_distances_across_features]
    avg_across_all = np.mean(trimmed, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(avg_across_all, marker='o', markersize=3, label="Average across all features & points")
    plt.title(f"{video_title}: Distance from First Frame (overall avg)")
    plt.xlabel("Frame")
    plt.ylabel("Average Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "overall_average.png"))
    plt.close()

    # Print Apex Frame Info
    apex_indices = np.where(avg_across_all == np.max(avg_across_all))[0]
    print("Apex Frame(s) (peak average distance from first frame):")
    for idx in apex_indices:
        timestamp = idx / frame_rate
        print(f"Frame {idx} at {timestamp:.2f} seconds (Distance={avg_across_all[idx]:.2f})")
    return avg_across_all

# --------- Main loop: extract frames at 24 FPS and process landmarks ---------

video_folder = "Videos"  # The directory containing all your video files

for filename in os.listdir(video_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(video_folder, filename)
        video_base = os.path.splitext(filename)[0]

        # ---- Define output folders for this video ----
        frames_dir = os.path.join("unalignedFrames_byVideo", video_base)   # Where raw frames will be stored
        graphs_dir = os.path.join("keypointGraphs_byVideo", video_base)    # Where plots/graphs will be saved
        npy_dir = os.path.join("numpy_landmarks_byVideo")                  # Where NumPy landmark arrays are stored
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)
        os.makedirs(npy_dir, exist_ok=True)

        # ---- STEP 1: Extract frames at exactly 24 FPS (ffmpeg) ----
        # This ensures you always get 24 frames per second, regardless of original video FPS.
        cmd = [
            "ffmpeg",
            "-i", input_path,                                 # Input video file
            "-vf", "fps=24",                                  # Set output to 24 frames per second
            os.path.join(frames_dir, "frame_%04d.jpg"),        # Naming convention for output frames
            "-hide_banner", "-loglevel", "error"              # Silence most ffmpeg output
        ]
        print(f"\nExtracting 24 FPS frames from {filename} ...")
        subprocess.run(cmd, check=True)
        print(f"Done extracting frames for {filename}.")

        # ---- STEP 2: Facial landmark extraction for each frame ----
        # Get list of all extracted frames, in order
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        print(f"Processing {len(frame_files)} frames for {filename} ...")

        # Initialize the face alignment model (loads deep learning weights once per video)
        fa = face_alignment.FaceAlignment(LandmarksType.TWO_D, flip_input=False, device='cpu')

        # Data structures for storing landmarks and trajectories
        all_grouped = []  # List of dictionaries (frame-by-frame facial feature points)
        feature_trajectories = {k: [] for k in FACIAL_FEATURES}  # Feature → list of points (per frame)

        # Loop over every frame, find facial landmarks, annotate & store results
        for idx, frame_name in enumerate(tqdm(frame_files, desc=f"Frames for {video_base}")):
            frame_path = os.path.join(frames_dir, frame_name)
            frame = cv2.imread(frame_path)

            grouped_frame = {}  # Dictionary to store all feature points for this frame

            preds = fa.get_landmarks(frame)  # Deep learning face landmark detection

            if preds is not None:
                # If landmarks found: extract and color-code points by facial feature
                keypoints = preds[0]  # (68,2) numpy array
                for feature, indices in FACIAL_FEATURES.items():
                    points = keypoints[indices].tolist()
                    grouped_frame[feature] = points
                    feature_trajectories[feature].append(points)
                    # Draw landmark points on frame for visualization
                    for (x, y) in points:
                        color = FEATURE_COLORS[feature]
                        cv2.circle(frame, (int(x), int(y)), 2, color, -1)
            else:
                # If no face detected, fill with NaN for this frame
                for feature, indices in FACIAL_FEATURES.items():
                    points = [[np.nan, np.nan]] * len(indices)
                    grouped_frame[feature] = points
                    feature_trajectories[feature].append(points)

            # Store result for this frame
            all_grouped.append(grouped_frame)
            # Save annotated image (overwrites frame)
            cv2.imwrite(frame_path, frame)

        # ---- STEP 3: Save all facial landmarks as .npz (NumPy) file ----
        # One file per video, each feature as an array of shape (num_frames, num_points, 2)
        landmarks_np = {feature: np.array(pts) for feature, pts in feature_trajectories.items()}
        np.savez(os.path.join(npy_dir, f"{video_base}_landmarks.npz"), **landmarks_np)

        # ---- STEP 4: Compute distance of each landmark point to its value in the first frame ----
        # Used to plot movement of face/features over time
        diff_from_first = {}
        for feature, frames_points in feature_trajectories.items():
            frames_points = np.array(frames_points)
            if len(frames_points.shape) != 3:
                continue  # Skip if not a sequence of points
            num_frames, num_points, _ = frames_points.shape
            diff_from_first[feature] = {}
            for pt_idx in range(num_points):
                ref = frames_points[0, pt_idx]  # Landmark coords in first frame
                diffs = []
                for frame_idx in range(num_frames):
                    coord = frames_points[frame_idx, pt_idx]
                    # If missing data, record as NaN
                    if np.any(np.isnan(coord)) or np.any(np.isnan(ref)):
                        d = float('nan')
                    else:
                        d = float(np.linalg.norm(coord - ref))
                    # Store [x, y, distance]
                    diffs.append([float(coord[0]), float(coord[1]), d])
                diff_from_first[feature][str(pt_idx)] = diffs

        # ---- STEP 5: Plot all feature movement graphs & print apex (max movement) info ----
        # Use frame_rate = 24 since all frames are extracted at 24 fps
        overall_avg = plot_feature_distances(diff_from_first, graphs_dir, video_base, 24.0)

print("\nAll done!")
