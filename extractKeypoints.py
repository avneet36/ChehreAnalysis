import face_alignment                      
from face_alignment import LandmarksType
import cv2                                 
import numpy as np
from tqdm import tqdm                      
import os                                  
import subprocess   
import sys          
import matplotlib.pyplot as plt
             

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

def getKeypoints(video_folder):
    numVids = 0
    limit = 1
    for filename in os.listdir(video_folder):
        if numVids >= limit:
            break
        if filename.endswith(".mp4"):
            input_path = os.path.join(video_folder, filename)
            video_base = os.path.splitext(filename)[0]

            # Define output folders for this video 
            frames_dir = os.path.join("framesByVideo", video_base)  # Where raw frames will be stored
            npy_dir = os.path.join("landmarksByVideo")              # Where NumPy landmark arrays are stored
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(npy_dir, exist_ok=True)

            # STEP 1: Extract frames at exactly 24 FPS (ffmpeg) 
            # This ensures you always get 24 frames per second, regardless of original video FPS.
            cmd = [
                "ffmpeg",
                "-i", input_path,                           # Input video file
                "-vf", "fps=24",                            # Set output to 24 frames per second
                os.path.join(frames_dir, "frame_%04d.png"), # Naming convention for output frames
                "-hide_banner", "-loglevel", "error"        # Silence most ffmpeg output
            ]
            print(f"\nExtracting 24 FPS frames from {filename} ...")
            subprocess.run(cmd, check=True)
            print(f"Done extracting frames for {filename}.")

            # STEP 2: Facial landmark extraction for each frame 
            # Get list of all extracted frames, in order
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
            print(f"Processing {len(frame_files)} frames for {filename} ...")

            # Initialize the face alignment model (loads deep learning weights once per video)
            fa = face_alignment.FaceAlignment(LandmarksType.TWO_D, flip_input=False, device='cpu')

            # Data structures for storing landmarks and trajectories
            all_grouped = []  # List of dictionaries (frame-by-frame facial feature points)
            feature_trajectories = {k: [] for k in FACIAL_FEATURES}  # Feature → list of points (per frame)
            all_keypoints = []  # Store all 68 keypoints for each frame

            # Loop over every frame, find facial landmarks, annotate & store results
            for idx, frame_name in enumerate(tqdm(frame_files, desc=f"Frames for {video_base}")):
                frame_path = os.path.join(frames_dir, frame_name)
                frame = cv2.imread(frame_path)
                
                # Get frame dimensions for normalization
                height, width = frame.shape[:2]
                grouped_frame = {}  # Dictionary to store all feature points for this frame
                preds = fa.get_landmarks(frame)  

                if preds is not None:
                    keypoints = preds[0]  # (68,2) numpy array
                    
                    # Store normalized keypoints for this frame
                    normalized_keypoints = keypoints.copy()
                    normalized_keypoints[:, 0] /= width   # normalize x coordinates
                    normalized_keypoints[:, 1] /= height  # normalize y coordinates
                    all_keypoints.append(normalized_keypoints)
                                
                    '''for feature, indices in FACIAL_FEATURES.items():
                        points = keypoints[indices].tolist()
                        grouped_frame[feature] = points
                        feature_trajectories[feature].append(points)
                        # Draw landmark points on frame for visualization
                        for (x, y) in points:
                            color = FEATURE_COLORS[feature]
                            cv2.circle(frame, (int(x), int(y)), 2, color, -1)'''
                else:
                    # If no face detected, fill with NaN for this frame
                    all_keypoints.append(np.full((68, 2), np.nan))
                    for feature, indices in FACIAL_FEATURES.items():
                        points = [[np.nan, np.nan]] * len(indices)
                        grouped_frame[feature] = points
                        feature_trajectories[feature].append(points)

                # Store result for this frame
                all_grouped.append(grouped_frame)
                # Save annotated image (overwrites frame)
                '''cv2.imwrite(frame_path, frame)'''

            # STEP 3: Save all facial landmarks as .npz (NumPy) file 
            # Single array of shape (num_frames, 68, 2) with normalized coordinates
            keypoints_array = np.array(all_keypoints) 
            np.savez(os.path.join(npy_dir, f"{video_base}_landmarks.npz"), keypoints=keypoints_array)
       
        numVids += 1
        
if __name__ == "__main__":
    video_folder = sys.argv[1]
    getKeypoints(video_folder)
    print("Facial keypoint extraction completed for all videos.")